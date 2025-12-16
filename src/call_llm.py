import json

import httpx
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Llama4ForConditionalGeneration, \
    BitsAndBytesConfig
from transformers import LlamaTokenizer, LlamaForCausalLM
import transformers
import torch
import os
import os
import json
import torch
import logging
import deepspeed
import torch.distributed as dist
import modelscope
from pathlib import Path
from prompt_toolkit import prompt
from typing import Optional, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import (login, HfFolder, snapshot_download,)
from openai import OpenAI


class GPT:
    def __init__(self, version):
        self.version = version
        api_key = ""
        self.client = OpenAI(api_key=api_key)

    def generate(self, messages):
        completion = self.client.chat.completions.create(
            model=self.version,
            messages=messages
        )
        answer = completion.choices[0].message.content
        if answer == "broken":
            print("Calling GPT failed, trying again.")
            self.generate(messages)
        return answer


class DeepSeek:
    def __init__(self, version):
        api_key = ""
        base_url = "https://api.deepseek.com"
        self.version = version  # model="deepseek-chat", "deepseek-reasoner"
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=httpx.Client(
                base_url=base_url,
                follow_redirects=True
            )
        )

    def generate_messages(self, messages):
        completion = self.client.chat.completions.create(
            model=self.version,
            messages=messages,
            stream=False
        )
        answer = completion.choices[0].message.content
        if answer == "broken":
            self.generate(prompt)
        return answer


class PsycheR1Chat:
    def __init__(
            self,
            model_name: str = "MindIntLab/Psyche-R1",
            model_path: Optional[str] = "",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            max_new_tokens: int = 2048,
            temperature: float = 0.7,
            top_p: float = 0.9,
            use_auth_token: Optional[str] = None,
            cache_dir: Optional[str] = None,
            local_rank: int = -1
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = model_name
        self.model_path = model_path
        self.cache_dir = cache_dir or Path.home() / ".cache" / "huggingface"
        self.local_rank = local_rank

        self.messages: List[Dict[str, str]] = []

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self._setup_auth(use_auth_token)
        self._setup_distributed()
        self._load_model_and_tokenizer()

    def _setup_distributed(self):
        if self.local_rank == -1:
            self.world_size = 1
            return

        deepspeed.init_distributed()
        torch.cuda.set_device(self.local_rank)
        self.world_size = torch.distributed.get_world_size()
        self.logger.info(
            f"The distributed environment is set up. Local process number: {self.local_rank}, total number of processes: {self.world_size}")

    def _setup_auth(self, use_auth_token: Optional[str]):

        if use_auth_token:
            login(token=use_auth_token)
            self.auth_token = use_auth_token
        elif os.getenv("HUGGINGFACE_TOKEN"):
            login(token=os.getenv("HUGGINGFACE_TOKEN"))
            self.auth_token = os.getenv("HUGGINGFACE_TOKEN")
        elif HfFolder.get_token():
            self.auth_token = HfFolder.get_token()
        else:
            self.logger.warning(
                "The HuggingFace authentication token was not provided. Please provide your token if you want to access private models."
            )
            self.auth_token = None

    def _load_model_and_tokenizer(self):

        try:
            if self.model_path and os.path.exists(self.model_path):
                self.logger.info(f"Loading model from local path: {self.model_path}...")
                model_path = self.model_path
            else:
                self.logger.info(f"Downloading model from HuggingFace {self.model_name}...")
                model_path = snapshot_download(
                    repo_id=self.model_name,
                    token=self.auth_token,
                    cache_dir=self.cache_dir,
                    local_files_only=False
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                token=self.auth_token if not self.model_path else None
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                token=self.auth_token if not self.model_path else None
            )

            model.eval()

            # DeepSpeed configs
            ds_inference_config = {
                "tensor_parallel": {
                    "tp_size": self.world_size
                },
                "dtype": "fp16",
                "replace_method": "auto",
                "replace_with_kernel_inject": True
            }

            self.logger.info(f"Current GPU: {self.local_rank}, total number of processes: {self.world_size}")
            self.logger.info(f"DeepSpeed configs: {ds_inference_config}")

            self.model = deepspeed.init_inference(
                model=model,
                config=ds_inference_config
            )

            self.logger.info(
                f"Model loaded successfully! Current process number: {self.local_rank}, total number of processes: {self.world_size}")

        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise

    def generate_response(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            **kwargs
    ) -> Dict[str, str]:
        # Update message history
        if system_prompt and not self.messages:
            self.messages.append({"role": "system", "content": system_prompt})
        self.messages.append({"role": "user", "content": prompt})

        try:
            # Formatting conversation history using message templates
            text = self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )

            input_ids = self.tokenizer.encode(text, return_tensors="pt")

            if self.local_rank != -1:
                input_ids = input_ids.to(f"cuda:{self.local_rank}")
            else:
                input_ids = input_ids.to(self.device)

            # generate responses
            with torch.no_grad():
                outputs = self.model.module.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            new_tokens = outputs[0][len(input_ids[0]):]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            self.messages.append({"role": "assistant", "content": response})

            return {"response": response}

        except Exception as e:
            self.logger.error(f"An error occurred while generating the response: {str(e)}")
            self.messages.pop()
            return {"error": str(e)}

    def generate_messages(self, messages):
        if self.local_rank in [-1, 0]:
            # print(
            #     "Welcome to Psyche-R1, the Chinese psychological reasoning LLM! Type "quit" or "exit" to end the conversation.")
            system_prompt = messages[0]["content"]
            # if system_prompt:
            #     print(f"\nSystem prompt: {system_prompt}\n")

            # while True:
            user_input = messages[1]["content"]
            # except (EOFError, KeyboardInterrupt):
            #     print("\n\nThanks for using the Chinese psychological reasoning LLM Psyche-R1. Goodbye!")
            #
            # if user_input.lower() in ["quit", "exit"]:
            #     print("\nThanks for using the Chinese psychological reasoning LLM Psyche-R1. Goodbye!")

            add_prefix_prompt = (
                # Chinese version
                # "你必须在<think>和</think>标签内给出你的推理过程，然后，在</think>标签后给出最终的答案。"
                # f"\n\n以下是我的问题：\n{user_input}"
                # English version
                "You need to provide your reasoning process within <think> and </think> tags, and then give your answer after the </think> tag."
                f"\n\nHere is my question:\n{user_input}"
            )

            try:
                response = self.generate_response(
                    add_prefix_prompt,
                    system_prompt=system_prompt if not self.messages else None
                )
                if "error" in response:
                    print(f"\nError: {response['error']}")
                else:
                    # print("\nAssistant:", response["response"])
                    return response["response"]
            except Exception as e:
                print(f"\nError occurred: {str(e)}")

    def clear_history(self):
        self.messages = []
        self.logger.info("Conversation history cleared.")


class Llama:
    def __init__(self, version):
        self.version = version
        if version == "Llama-3.1-8B-Instruct":
            model_dir = "meta-llama/Llama-3.1-8B-Instruct"
            model_raw = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
            self.model = model_raw.half()
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        elif version in ["Llama-3.3-70B-Instruct", "Llama-4-Scout", "DeepSeek-R1-Distill-Llama-70B"]:
            if version == "Llama-3.3-70B-Instruct":
                dir = "meta-llama/Llama-3.3-70B-Instruct"
            if version == "Llama-4-Scout":
                dir = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
            elif version == "DeepSeek-R1-Distill-Llama-70B":
                dir = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # 4位量化
                bnb_4bit_compute_dtype=torch.float16,  # 计算时使用FP16
                bnb_4bit_quant_type="nf4",  # 量化类型 (推荐nf4/fp4)
                bnb_4bit_use_double_quant=True  # 二次量化节省更多空间
            )
            model = AutoModelForCausalLM.from_pretrained(
                dir,
                quantization_config=quantization_config,  # 关键变更点
                device_map="auto",
                low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(dir)
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto"
            )

    def generate_messages(self, messages):
        device = "cuda"
        if self.version == "Llama-3.1-8B-Instruct":
            model = self.model
            tokenizer = self.tokenizer
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=2048
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        elif self.version in ["Llama-3.3-70B-Instruct", "Llama-4-Scout"]:
            # https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
            pipeline = self.pipeline
            outputs = pipeline(
                messages,
                max_new_tokens=2048,
            )
            return outputs[0]["generated_text"][-1]["content"]


class DSDistill:
    def __init__(self, version):
        self.version = version
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "deepseek-ai/" + self.version
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # 使用BF16精度节省显存
            device_map="auto",
            trust_remote_code=True
        ).to(self.device).eval()

    def generate_messages(self, messages):
        tokenizer = self.tokenizer
        device = self.device
        model = self.model
        tokenized_chat = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        outputs = model.generate(
            tokenized_chat,
            max_new_tokens=2048,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        response = tokenizer.decode(outputs[0][tokenized_chat.shape[1]:], skip_special_tokens=True)
        return response.strip()


class Qwen:
    def __init__(self, version):
        if "mindora_chord" in version:
            model_name = "elsashaw/mindora_chord"
        elif "mindora_rl" in version:
            model_name = "elsashaw/mindora_rl"
        elif "EmoLLM" in version:
            model_name = "aJupyter/EmoLLM_Qwen2-7B-Instruct_lora"
        else:
            model_name = "Qwen/" + version
        print("loading model in dir:", model_name)
        print("start loading self.model")
        self.model = modelscope.AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cuda"
        )
        print("finish loading self.model")
        print("start loading tokenizer")
        self.tokenizer = modelscope.AutoTokenizer.from_pretrained(model_name)
        print("finish loading tokenizer")

    def generate(self, prompt):
        messages = [
            {"role": "system", "content": "You are a psychologist."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def generate_messages(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response



class QwenApi:
    def __init__(self, version):
        self.client = OpenAI(
            api_key="",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.version = version

    def generate_messages(self, messages):
        if self.version == 'qwen2.5-72b-instruct':
            try:
                completion = self.client.chat.completions.create(
                    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                    model=self.version,
                    messages=messages,
                    # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
                    # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
                    # extra_body={"enable_thinking": False},
                )
                response = json.loads(completion.model_dump_json())
                response = response["choices"][0]["message"]["content"]
            except Exception as e:
                print(e)
                response = "content_filter"
            return response

        else:  # reasoning models
            try:
                completion = self.client.chat.completions.create(
                    model=self.version,  # 您可以按需更换为其它深度思考模型
                    messages=messages,
                    # enable_thinking 参数开启思考过程，QwQ 与 DeepSeek-R1 模型总会进行思考，不支持该参数
                    extra_body={"enable_thinking": True},
                    stream=True,
                    # stream_options={
                    #     "include_usage": True
                    # },
                )

                reasoning_content = ""  # 完整思考过程
                answer_content = ""  # 完整回复
                is_answering = False  # 是否进入回复阶段
                # print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

                for chunk in completion:
                    if not chunk.choices:
                        print("\nUsage:")
                        print(chunk.usage)
                        continue

                    delta = chunk.choices[0].delta

                    # 只收集思考内容
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                        # if not is_answering:
                        #     print(delta.reasoning_content, end="", flush=True)
                        reasoning_content += delta.reasoning_content

                    # 收到content，开始进行回复
                    if hasattr(delta, "content") and delta.content:
                        if not is_answering:
                            # print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                            is_answering = True
                        # print(delta.content, end="", flush=True)
                        answer_content += delta.content
            except Exception as e:
                print(e)
                return "", "content_filter"
            return reasoning_content, answer_content
