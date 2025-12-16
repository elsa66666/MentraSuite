import json
import os
import argparse
import time

from call_llm import *
import re
from statistics import mean
from check_answer import *


def cognitive_error_list(dataset):
    if dataset == 'CognitiveReframing':
        # https://arxiv.org/pdf/2305.02466
        error_prompt_list = (
                "\n[All-or-Nothing Thinking]: Thinking in extremes." +
                "\n[Overgeneralization]: Jumping to conclusions based on one experience." +
                "\n[Labeling]: Defining a person based on one action or characteristic." +
                "\n[Fortune telling]: Trying to predict the future. Focusing on one possibility and ignoring other, more likely outcomes." +
                "\n[Mind reading]: Assuming you know what someone else is thinking. " +
                "\n[Emotional Reasoning]: Treating your feelings like facts." +
                "\n[Should statements]: Setting unrealistic expectations for yourself." +
                "\n[Personalization]: Taking things personally or making them about you." +
                "\n[Discounting the Positive]: When something good happens, you ignore it or think it doesn't count. " +
                "\n[Catastrophizing]: Focusing on the worst-case scenario." +
                "\n[Comparing and Despairing]: Comparing your worst to someone else's best." +
                "\n[Blaming]: Giving away your own power to other people. " +
                "\n[Negative Feeling or Emotion]: Getting stuck on a distressing thought, emotion, or belief."
        )
        error_list = ['All-or-Nothing Thinking', 'Overgeneralization', 'Labeling', 'Fortune telling',
                      'Mind reading', 'Emotional Reasoning', 'Should statements', 'Personalization',
                      'Discounting the Positive', 'Catastrophizing', 'Comparing and Despairing',
                      'Blaming', 'Negative Feeling or Emotion']
    elif dataset == 'PatternReframe':
        # https://arxiv.org/pdf/2307.02768
        error_prompt_list = (
                "\n[Catastrophizing]: by giving greater weight to the worst possible outcome. " +
                "\n[Discounting the positive]: refers to when individuals achieve positive outcomes but attribute these successes to luck or believe that anyone could achieve them, thus negating their own effort and value. " +
                "\n[Overgeneralization]: can be only used to generalize a particular event leading to a certain outcome to all similar events. " +
                "\n[Personalization]: assigning a disproportionate amount of personal blame to oneself. " +
                "\n[All-or-Nothing Thinking]: viewing things as either good or bad and nothing in between. " +
                "\n[Mental filtering]: an event has both positive and negative aspects, but individuals focus only on the negative aspects." +
                "\n[Mind reading]: inferring a person's probable (usually negative) thoughts from their behavior. " +
                "\n[Fortune telling]: is predicting outcomes of events. " +
                "\n[Should statements]: where a person demands particular behaviors regardless of the realistic circumstances. " +
                "\n[Labeling]: attributing a person's actions to their character. "
        )
        error_list = ['Catastrophizing', 'Discounting the positive', 'Overgeneralization', 'Personalization',
                      'All-or-Nothing Thinking', 'Mental filtering', 'Mind reading', 'Fortune telling',
                      'Should statements', 'Labeling']
    elif dataset == 'therapistQA':
        # https://aclanthology.org/2021.clpsych-1.17.pdf
        error_prompt_list = (
                "\n[Emotional Reasoning]: Believing 'I feel that way, so it must be true'." +
                "\n[Overgeneralization]: Drawing conclusions with limited and often un negative experience." +
                "\n[Mental Filtering]: Focusing only on limited negative aspects and not the excessive positive ones." +
                "\n[Should Statements]: Expecting things or personal behavior should be a certain way." +
                "\n[All or Nothing thinking]: Binary thought pattern. Considering anything short of perfection as a failure." +
                "\n[Mind Reading]: Concluding that others are reacting negatively to you, without any basis in fact." +
                "\n[Fortune Telling]: Predicting that an event will always result in the worst possible outcome." +
                "\n[Magnification]: Exaggerating or Catastrophizing the outcome of certain events or behavior." +
                "\n[Personalization]: Holding oneself personally responsible for events beyond one's control." +
                "\n[Labeling]: Attaching labels to oneself or oth-ers (ex: 'loser', 'perfect'). " +
                "\n[No Distortion]: The client doesn't have cognitive error."
        )
        error_list = ['Emotional Reasoning', 'Overgeneralization', 'Mental Filtering', 'Should Statements',
                      'All or Nothing thinking',
                      'Mind Reading', 'Fortune Telling', 'Magnification', 'Personalization', 'Labeling',
                      'No Distortion']
    return error_prompt_list, error_list


def get_dataset_path_and_save_path():
    save_root_path = "../result/"+args.llm+"/"
    save_path = save_root_path + args.dataset_name + "_" + args.llm + ".json"
    # appraisal
    if args.dataset_name == 'CognitiveReframing':
        dataset_path = "../data/test/CognitiveReframing_test_284.json"
    elif args.dataset_name == 'PatternReframe':
        dataset_path = "../data/test/PatternReframe_test_300.json"
    elif args.dataset_name == 'therapistQA':
        dataset_path = "../data/test/therapistQA_test_507.json"
    # diagnosis
    elif args.dataset_name == 'DepSign':
        dataset_path = "../data/test/DepSign_test_600.json"
    elif args.dataset_name == "swmh":
        dataset_path = "../data/test/swmh_test_500.json"
    elif args.dataset_name == "tsid":
        dataset_path = "../data/test/tsid_test_942.json"
    # intervention
    elif args.dataset_name == "annomi":
        dataset_path = "../data/test/annomi_test_133.json"
    # multi-steps
    elif args.dataset_name == "mhqa":
        dataset_path = "../data/test/mhqa_test_717.json"
    elif args.dataset_name == "medqa":
        dataset_path = "../data/test/medqa_test_121.json"
    elif args.dataset_name == "medmcqa":
        dataset_path = "../data/test/medmcqa_test_446.json"
    elif args.dataset_name == "pubmedqa":
        dataset_path = "../data/test/pubmedqa_test_89.json"
    # abstraction
    elif args.dataset_name == "PSRS":
        dataset_path = "../data/test/PSRS_test_108.json"
    # verification
    elif args.dataset_name == "misinfo":
        dataset_path = "../data/test/misinfo_test_130.json"
    return dataset_path, save_path


def get_case_info(case):
    context = case['query']
    dataset_name = args.dataset_name
    if dataset_name != "PSRS":
        index = case["index"]
        reference_answer = case['label']
    else:
        index = case["id"]
        reference_answer = case["reference_answer"]
    if dataset_name in ['mhqa','medmcqa','medqa','pubmedqa']:
        if dataset_name == "medmcqa":
            question = "Question: Please choose the most proper option from: a, b, c, d."
        elif dataset_name == "medqa":
            question = "Question: Please choose the most proper option from: a, b, c, d, e."
        elif dataset_name == "mhqa":
            question = "Question: Please choose the most proper option from: a, b, c, d."
        elif dataset_name == "pubmedqa":
            question = "Question: Please choose the most proper option from: yes, no, maybe."
    elif dataset_name == 'annomi':
        skills = '''
                [Clarification]:
                Clarification is a form of questioning that serves to get a clearer understanding of what a client has said. It is often used at the beginning of counseling or when starting a new topic. The purpose is to encourage the client to elaborate, to confirm the accuracy of information, and to clear up ambiguous or confusing messages.
                There are four steps to applying clarification. First, attend to the client's verbal and non-verbal messages. Second, identify any vague or confusing information that needs to be checked. Third, choose an appropriate opening, using a question format. Fourth, assess the effectiveness of the clarification by listening to and observing the client's expression and response.

                [Paraphrasing]:
                Paraphrasing is a restatement of the content portion of the client's message. Its purpose is to help the client focus on the information they have shared and to highlight the content of their message, especially when they are prematurely focusing on emotions or engaging in self-criticism.
                There are four steps to applying a paraphrase. First, recall the information provided by the client. Second, identify the content portion of the message. Third, use your own words to restate the main content or concepts of the client's message, ensuring you use a statement tone. Fourth, confirm its effectiveness by listening to and observing the client's expression and response.  
                Beginners often confuse clarification and paraphrasing. Their two main differences are:
                Clarification uses an inquisitive tone, while paraphrasing uses a statement tone.
                Clarification often uses the client's own language, while paraphrasing uses the counselor's language.

                [Reflection of Feeling]:
                Reflection of feeling is a restatement of the emotional part of a client's message.
                Generally, in the early stages of counseling, reflection of feeling should be used cautiously. Overuse can make the client uncomfortable and lead them to deny their emotional experiences. However, in the later stages of counseling, after a good therapeutic relationship has been established, focusing on the client's emotional responses can greatly facilitate the counseling process.
                The purpose of reflecting feelings is to encourage the client to express more of their feelings, help them experience intense emotions, help them realize they can have agency over their emotions, help them recognize and manage their emotions, and help them identify their emotions.

                There are four steps to applying reflection of feeling. First, listen for the emotion words the client uses. Second, pay close attention to the client's non-verbal information, such as body posture, facial expressions, and vocal characteristics. Non-verbal behavior is harder to control than verbal behavior and is a more reliable indicator of emotion. Third, choose appropriate words to reflect the identified emotion back to the client, matching both the type and intensity level of the emotion. Sometimes, you can add the context of the situation before the feeling reflection. Fourth, assess whether the reflection of feeling was effective by determining if the client agrees with the counselor's reflection.

                [Summarizing]:
                Summarizing is condensing the client's messages using two or more paraphrases or reflections of feeling. A summary can be seen as a reflection of the content and/or feeling themes in a client's message, or a combination of both. It can be used after just a few minutes of conversation or after several sessions, primarily determined by the needs of the session and the purpose of the summary.
                The purpose of summarizing is to connect multiple elements within the client's messages, identify a common theme or pattern, interrupt redundant statements, and review the overall process.

                There are four steps to applying a summary. First, recall the information the client has conveyed, both verbal and non-verbal. Second, identify any clear patterns, themes, or multiple elements present in the client's messages. Third, choose an appropriate opening to begin the summary, using "you" or the client's name. Summarize and restate the themes the client expressed using the counselor's own words, being sure to use a statement tone rather than a questioning one. Fourth, assess whether the summary was effective and if the client agrees with it.

                [Questioning Skills]:
                Open-ended Questions (the counselor has no preconceived answer, and the client cannot give a simple reply; used to gather information) and Closed-ended Questions (the counselor has a preconceived answer, and the client does not need to elaborate; used to clarify a point).

                [Immediacy]:
                A verbal response to what is happening in the here-and-now of the counseling interview.

                [Use of Silence]:
                During the counseling process, the counselor's appropriate use of and response to the client's silence, or the intentional creation of silence, to provide the client with sufficient time and space for self-reflection.

                [Self-Disclosure]:
                There are generally two forms. One is when the counselor shares their own experiences and feelings about the client with the client. The other is disclosing personal past experiences relevant to the topic the client is discussing.

                [Confrontation]:
                Pointing out contradictions or inconsistencies present in the client. Common contradictions include: 1. Inconsistency between ideals and reality. 2. Inconsistency between words and actions. 3. Inconsistency between previous and current statements.

                [Encouragement]:
                Using verbal cues to encourage the client, prompting them to engage in self-exploration or to guide a topic shift, by selectively attending to a specific aspect of what the client is saying.

                [Repetition]:
                Directly repeating a specific phrase the client has just stated to draw their attention to that statement and to what it is intended to express.

                [Interpretation]:
                Using psychological theory to explain the reasons for and nature of a client's thoughts, feelings, and behaviors, or to explain certain abstract and complex psychological processes.

                The difference with paraphrasing is that paraphrasing explains the essential content of the client's expression from the client's frame of reference, whereas interpretation uses the counselor's frame of reference, applying psychological theory and life experience to provide a theoretical analysis of an issue.

                [Guidance]:
                The counselor directly instructs the client to do something, say certain things, or act in a particular way.

                '''
        question = f"According to this document:{skills}. For the following visitor text, select one or more correct counseling strategies. If there are multiple ones, please use them and separate them like [Clarification,Interpretation]."
    elif dataset_name in ['CognitiveReframing','PatternReframe','therapistQA']:
        error_prompt_list, _ = cognitive_error_list(dataset_name)
        question = "Question: Please choose the most proper cognitive error of this client from the following list: " + error_prompt_list
    elif dataset_name == 'swmh':
        question = "Question: Please choose the most proper mental disorder of this poster from: suicide, anxiety, bipolar disorder, depression, none. Where 'none' means the poster shows no sign of mental disorder."
    elif dataset_name == 'tsid':
        question = "Question: Please choose the most proper mental disorder of this poster from: suicide, depression, ptsd, none. Where 'none' means the poster shows no sign of mental disorder."
    elif dataset_name == 'DepSign':
        question = "Question: Please choose the most proper depression degree of this poster from: 0,1,2. Where 0 for not depressed, 1 for moderately depressed, and 2 for severely depressed."
    elif dataset_name == 'misinfo':
        question = "Question: Does the video content contain mental health misinformation? Options: yes, no."
    elif dataset_name == "PSRS":
        question = "Please summarize the main findings by point from the 'Main results' section of this abstract, reply in short."
    context = "Context: " + context
    return index, question, context, reference_answer


def get_llm():
    if args.llm in ['o4-mini', 'gpt-4o']:
        llm = GPT(version=args.llm)
    elif args.llm in ['deepseek-chat', 'deepseek-reasoner']:
        llm = DeepSeek(version='deepseek-chat')
    if args.llm in ['qwen-plus', 'qwq-plus', 'qwen2.5-72b-instruct', 'llama-4-scout-17b-16e-instruct', 'deepseek-r1-distill-llama-70b', 'deepseek-r1-distill-qwen-32b', 'qwen3-32b', 'qwq-32b',
                    'deepseek-v3', 'deepseek-r1']:
        llm = QwenApi(version=args.llm)
        return llm
    if args.llm in ["Llama-3.3-70B-Instruct","Llama-3.1-8B-Instruct"]:
        llm = Llama(version=args.llm)
    elif 'DeepSeek-R1-Distill' in args.llm:
        llm = DSDistill(version=args.llm)
    elif args.llm in ["Qwen3-8B", "Qwen3-14B", "Qwen3-32B", "EmoLLM"]:
        llm = Qwen(version=args.llm)
    elif "mindora" in args.llm:
        llm = Qwen(version=args.llm)
    elif args.llm == 'psyche_r1':
        llm = PsycheR1Chat()
    return llm


def test_llm_reasoning():
    dataset_path, save_path = get_dataset_path_and_save_path()
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    llm = get_llm()
    result_list = []
    check_list = []
    for i in range(args.start_num, len(test_data)):
        case = test_data[i]
        index, question, context, reference_answer = get_case_info(case)
        print('--------start toggling', index, ',', i, '/', len(test_data), '--------')
        messages = [
            {
                "role": "system",
                "content": "Based on the following information of a case to make judgements. When answering, follow these steps concisely:\n\n 1. Reasoning Phase:\n   - Enclose all analysis within <think> tags\n   - Use structured subtitles (e.g., '###Comparing with Given Choices:') on separate lines\n   - Final section must be '###Final Conclusion:'\n\n2. Answer Phase:\n - Enclose your answer within <answer> tags\n - The answer phase should end with 'Answer: [option]'.\n - The answer should be aligned with reasoning phase. \nDeviation from this format is prohibited."
            },
            {
                "role": "user",
                "content": question + "\n" + context + "\n"
            }
        ]
        # print(messages)
        if args.llm  == "llama-4-scout-17b-16e-instruct":
            if count == 10:
                time.sleep(60)
                count = 0
            reasoning, response = llm.generate(prompt)
        original_response = llm.generate_messages(messages)
        if args.llm == "psyche_r1":
            # for Psyche-r1
            llm.clear_history()
        print("---> original response: ", original_response)
        match1 = re.search(r'<answer>(.*?)</answer>', original_response, re.DOTALL)
        if match1:
            generated_answer = match1.group(1).strip()
        else:
            match11 = re.search(r'<well-structured-response>(.*?)</well-structured-response>', original_response, re.DOTALL)
            if match11:
                generated_answer = match11.group(1).strip().split("\n")[-1]
            else:
                generated_answer = original_response.split("\n")[-1]
        match2 = re.search(r'<think>(.*?)</think>', original_response, re.DOTALL)
        if match2:
            reasoning = match2.group(1)
        else:
            reasoning = "".join(original_response.split("\n")[:-1])
        print("---> generated answer from response regex:", generated_answer)
        # print("reasoning:", reasoning)
        if args.dataset_name != "PSRS":
            generated_label, check1 = check(index, generated_answer, reference_answer)
            print("---> generated label:", generated_label)
            print("---> reference label:", reference_answer)
            dict1 = {
                "index": index,
                "question": question,
                "context": context,
                "original_response": original_response,
                "reasoning": reasoning,
                "generated_answer": generated_answer,
                "generated_label": generated_label,
                "reference_label": reference_answer,
                "check": check1
            }
        else:
            scoring_reason, check1 = check_PSRS(index, original_response, reference_answer)
            print("---> scoring reason:", scoring_reason)
            dict1 = {
                "index": "PSRS_test_" + str(i),
                "id": index,
                "question": question,
                "context": context,
                "original_response": original_response,
                "reasoning": reasoning,
                "generated_answer": generated_answer,
                "reference_label": reference_answer,
                "scoring_reason": scoring_reason,
                "check": check1
            }
        print("---> check:", check1)

        result_list.append(dict1)
        check_float = process_value(check1)
        check_list.append(check_float)

        with open(save_path, "w", encoding="utf-8") as json_file:
            json.dump(result_list, json_file, ensure_ascii=False, indent=4)
    average_score = mean(check_list)
    print('score:', round(average_score, 4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, options=[
        "o4-mini", "gpt-4o",  # gpt api
        "deepseek-chat", "deepseek-reasoner", # to use Deepseek api for ds-r1 and ds-v3
        "deepseek-v3", "deepseek-r1",  # to use qwen-api for ds-r1 and ds-v3
        "qwen-plus", "qwq-plus", # qwen-api
        "Llama-3.3-70B-Instruct", # llama local
        "llama-4-scout-17b-16e-instruct", "deepseek-r1-distill-llama-70b", "qwen2.5-72b-instruct",  # qwen-api
        "deepseek-r1-distill-qwen-32b", "qwen3-32b", "qwq-32b", # qwen-api
        "DeepSeek-R1-Distill-Qwen-14B", # ds-distill local
        "Qwen3-14B",  # qwen local
        "Llama-3.1-8B-Instruct", # llama local
        "DeepSeek-R1-Distill-Llama-8B",  # ds-distill local
        "EmoLLM", "psyche_r1", "Qwen3-8B", # qwen local
        "mindora_chord", "mindora_rl" # qwen local
    ])
    parser.add_argument("--dataset_name", type=str, options=[
        "CognitiveReframing", "PatternReframe", "therapistQA",
        "DepSign", "swmh", "tsid",
        "annomi",
        "mhqa", "medqa", "medmcqa", "pubmedqa",
        "PSRS",
        "misinfo"
    ])
    parser.add_argument("--start_num", type=int, default=0)
    args = parser.parse_args()
    if not os.path.exists('../result/'+args.llm+'/'):
        os.mkdir('../result/'+args.llm+'/')
    test_llm_reasoning()
