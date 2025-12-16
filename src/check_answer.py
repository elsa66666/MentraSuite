from call_llm import *
import json_repair


def check_answer_AnnoMI(generated_reply, reference_answer):
    # 你的判断代码
    # print(f'generated_reply={generated_reply}')
    # print(f'reference_answer={reference_answer}')
    # generate_label_list = str(generated_reply[1:-1]).split(',')
    # print(f'generate_label_list={generate_label_list}')
    # reference_label_list = str(reference_answer).split('、')
    # print(f'reference_label_list={reference_label_list}')
    skills_list = ["Clarification", "Paraphrasing", "Reflection of Feeling", "Summarizing", "Questioning Skills",
                   "Immediacy", "Use of Silence", "Self-Disclosure", "Confrontation", "Encouragement", "Repetition",
                   "Interpretation", "Guidance"]
    generate_label_list = []
    for skill in skills_list:
        if skill.lower() in generated_reply.lower():
            generate_label_list.append(skill)
    reference_label_list = []
    for skill in skills_list:
        if skill.lower() in reference_answer.lower():
            reference_label_list.append(skill)

    # generated_label_true = []  # 筛查
    # for label in generate_label_list:
    #     label = label.strip()
    #     if label in skills_list:
    #         generated_label_true.append(label)
    # reference_label_true = []
    # for label in reference_label_list:
    #     label = label.strip()
    #     if label in skills_list:
    #         reference_label_true.append(label)
    # generated_label = ', '.join(generated_label_true)
    # reference_label = ', '.join(reference_label_true)
    # print(f'After check generated_label={generated_label}')
    # print(f'After check reference_label={reference_label}')
    # # 算check
    # generated_label_set = set([str(x).lower() for x in generated_label_true if isinstance(x, str)])
    # reference_label_set = set([str(x).lower() for x in reference_label_true if isinstance(x, str)])
    generated_label_set = set(generate_label_list)
    reference_label_set = set(reference_label_list)
    # 计算交集和并集
    intersection = generated_label_set.intersection(reference_label_set)
    union = generated_label_set.union(reference_label_set)

    # 计算交集和并集的个数
    intersection_count = len(intersection)
    union_count = len(union)

    # 计算比值
    if union_count > 0:
        ratio = intersection_count / union_count
    else:
        ratio = 0
    print(f"check：{ratio}")
    generated_label = ",".join(generate_label_list)
    reference_label = ",".join(reference_label_list)
    return generated_label, reference_label, ratio


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


def check_answer_cr(generated_reply, reference_answer, error_list):
    generated_reply = generated_reply.split('\n')[-1]
    check = False
    generated_label = ''
    for err in error_list:
        if err.lower().replace('-', ' ') in generated_reply.lower().replace('-', ' '):
            generated_label = err
            break

    reference_list = []
    for err in error_list:
        if err.lower().replace('-', ' ') in reference_answer.lower().replace('-', ' '):
            reference_list.append(err)

    # for reference_label in reference_answer:  # 答对一个就算对
    #     reference_label = reference_label.strip()
    #     if generated_reply != '':
    #         if reference_label.replace('-', ' ').lower() in generated_reply.replace('-', ' ').lower():
    #             check = True
    # for err in error_list:
    #     if err.replace('-', ' ').lower() in generated_reply.replace('-', ' ').lower():
    #         generated_label = err
    if generated_label != '' and generated_label in reference_list:
        check = True
    return generated_label, check


def check_answer_crisis(dataset, generated_reply, reference_answer):
    generated_label = generated_reply
    if dataset in ['dreaddit', 'irf', 'loneliness']:
        if 'yes' in generated_label:
            generated_label = "yes"
        elif "no" in generated_label:
            generated_label = "no"
        else:
            generated_label = ""

    elif dataset == 'swmh':
        matched = False
        for opt in ['suicide', 'anxiety', 'bipolar disorder', 'depression', 'none']:
            if opt in generated_label:
                generated_label = opt
                matched = True
        if not matched:
            generated_label = ""

    elif dataset == 'tsid':
        matched = False
        for opt in ['suicide', 'depression', 'ptsd', 'none']:
            if opt in generated_label:
                generated_label = opt
                matched = True
        if not matched:
            generated_label = ""

    elif dataset == 'DepSign':
        matched = False
        for opt in ['0', '1', '2']:
            if opt in generated_label:
                generated_label = opt
                matched = True
        if not matched:
            generated_label = ""

    if str(generated_label) == str(reference_answer):
        check = True
    else:
        check = False

    return generated_label, check


def check_answer_diagnosis(generated_reply, reference_answer):
    generated_label = generated_reply

    matched = False
    answer_list = []
    for opt in [str(i) for i in range(1, 20)]:
        if opt in str(generated_label):
            answer_list.append(opt)
            matched = True
    if not matched:
        generated_label = ""
    check = False
    for answer_opt in answer_list:
        if str(answer_opt) in str(reference_answer):
            check = True
    return generated_label, check


def checkQA(generated_answer, reference_answer):
    generated_label = ""
    check = False
    matched = False
    option1_list = ["a.", "b.", "c.", "d.", "e."]
    for option in option1_list:
        if option in generated_answer and not matched:
            generated_label = option[0]
            matched = True
    if not matched:
        option2_list = ["a", "b", "c", "d", "e"]
        for option in option2_list:
            if option in generated_answer and not matched:
                generated_label = option
                matched = True
    if not matched:
        generated_label = "manual check"
    if generated_label == reference_answer:
        check = True
    return generated_label, check


def check(index, generated_answer, reference_label):
    if "answer:" in generated_answer.lower():
        generated_answer = generated_answer.lower().split("answer:")[-1]
        print("---> extracted substring after 'answer':", generated_answer)

    index = index.replace("-", "_")
    dataset = index.split("_")[0]
    print("---> dataset:", dataset)
    generated_label = ""
    check = False
    if dataset in ["MedMCQA","MedQA", "mhqa"]:
        generated_label, check = checkQA(generated_answer, reference_label)

    elif "PubMedQA" in index:
        option_list = ["yes", "no", "maybe"]
        matched = False
        for option in option_list:
            if option in generated_answer and not matched:
                generated_label = option
                matched = True
        if not matched:
            generated_label = "manual check"
        if generated_label == reference_label:
            check = True
        else:
            check = False

    elif "AnnoMI" in index:
        generated_label, _, check = check_answer_AnnoMI(generated_answer, reference_label)

    elif dataset in ["CognitiveReframing","PatternReframe","therapistQA"]:
        dataset = index.split("_")[0]
        _, error_list = cognitive_error_list(dataset)
        # reference_list = list(set(reference_label.lower().split(',')))
        generated_label, check = check_answer_cr(generated_answer, reference_label, error_list)

    elif dataset in ["DepSign","swmh","tsid"]:
        generated_label, check = check_answer_crisis(dataset, generated_answer, reference_label)

    elif "misinfo" in index:
        generated_label = generated_answer.strip()
        if generated_label == reference_label:
            check = True
        else:
            check = False

    elif index == "check_knowledge":  # 这种情况下，标答是incorrect，看大模型能否发现错误
        if "incorrect" in generated_answer:
            generated_label = "incorrect"
            check = True
        else:
            check = False
    return generated_label, check


def check_PSRS(index, generated_answer, reference_answer):
    llm = GPT(version="o1-mini")
    prompt = (
                "Evaluate the generated response against the reference answer points. "
                "Identify how many reference points are correctly addressed and report both the matched count and the total number of reference points. \n"
                f"Generated Response: {str(generated_answer)}\n"
                f"Reference Answer: {str(reference_answer)}\n"
                "Return the result in JSON format as: "
                "{\"correct_points\": number, \"total_points\": number, \"reason\": \"one-sentence explanation\" }"
            )

    # 调用 LLM 直到获取有效响应
    parsed_response = ""
    while not parsed_response:
        try:
            response = llm.generate(prompt)
            parsed_response = json_repair.repair_json(response, return_objects=True)
            # 验证响应格式（确保包含必要字段）
            required_fields = ['correct_points', 'total_points', 'reason']
            if not all(field in parsed_response for field in required_fields):
                print(f"警告：LLM 响应缺少必要字段，重新请求（index: {index}）")
                parsed_response = ""
        except Exception as e:
            print(f"LLM 调用失败（index: {index}）：{str(e)}，重试中...")
            parsed_response = ""

    # 计算分数
    try:
        score = round(parsed_response['correct_points'] / parsed_response['total_points'], 4)
    except ZeroDivisionError:
        print(f"警告：index {index} 的 total_points 为 0，分数设为 0")
        score = 0.0
    scoring_reason = str(parsed_response['correct_points']) + "/" + str(parsed_response['total_points']) + parsed_response["reason"]
    return scoring_reason, score


def process_value(check):
    if isinstance(check, bool):
        # 布尔值转换为1.0或0.0
        return 1.0 if check else 0.0
    elif isinstance(check, float):
        # 浮点数保留两位小数
        return check
