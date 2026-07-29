from eff.data_model.simulation import RewardInfo, EvaluationType, SimulationRun, TerminationReason
from eff.data_model.tasks import Task
from eff.evaluator.evaluator_traj import TrajectoryEvaluator


def evaluate_simulation(
    simulation: SimulationRun,
    task: Task,
    evaluation_type: EvaluationType,
    domain: str,
    llm_evaluator: str = None,
    llm_args_evaluator: dict = None,
    language: str = None,
) -> RewardInfo:
    """
    对仿真运行结果进行评估，根据评估类型选择不同的 reward 计算策略，返回 RewardInfo。

    整体流程：
    1. 首先检查仿真是否因异常原因提前终止（错误过多 / 超过最大步数 / 无效消息），
       如果是则直接返回 reward=0.0，不再调用 LLM 评估。
    2. 如果任务没有定义评估标准（evaluation_criteria 为 None），视为通过，返回 reward=1.0。
    3. 根据 evaluation_type 选择对应的 reward 计算方法：
       - "trajectory"：滑动窗口 + rubric 评估（默认方式）
       - "trajectory_full_traj_rubric"：全轨迹 + rubric 评估
       - "trajectory_sliding_wo_rubric"：滑动窗口 + 无 rubric（LLM 自行判断）
       - "trajectory_full_traj_wo_rubric"：全轨迹 + 无 rubric

    Args:
        simulation: 仿真运行结果，包含完整对话轨迹 (messages) 和终止原因等信息。
        task: 任务定义，包含评估标准 (evaluation_criteria)、用户指令等。
        evaluation_type: 评估类型，决定使用哪种 reward 计算策略。
        domain: 业务领域（ridehailing）。
        llm_evaluator: 用于评估的 LLM 模型名称，为 None 时使用默认模型。
        llm_args_evaluator: 传递给评估 LLM 的额外参数。
        language: 评估 prompt 的语言（如 "chinese"）。

    Returns:
        RewardInfo: 包含以下字段：
            - reward (float): 最终奖励分数，0.0 或 1.0（全部 rubric 通过才为 1.0）。
            - nl_rubrics (list[NLRubricCheck]): 每条 rubric 的评估结果（是否通过及理由）。
            - reward_breakdown (dict): 按 RewardType 维度分解的分数。
            - info (dict): 附加评估信息（评估方法、窗口数量等）。
    """
    # 1. 异常终止检查：如果仿真因错误过多、超过最大步数或无效消息而终止，直接返回 0 分
    if simulation.termination_reason in {
        TerminationReason.TOO_MANY_ERRORS,
        TerminationReason.MAX_STEPS,
        TerminationReason.INVALID_AGENT_MESSAGE,
    }:
        return RewardInfo(
            reward=0.0,
            info={
                "note": f"Simulation terminated prematurely. Termination reason: {simulation.termination_reason}"
            },
        )
    # 2. 无评估标准时，视为通过
    if task.evaluation_criteria is None:
        return RewardInfo(
            reward=1.0,
            info={"note": "No evaluation criteria"},
        )
    # 3. 根据评估类型分派到不同的 reward 计算方法
    if evaluation_type == "trajectory":
        # 自适应策略：对话 ≤ 80 条消息时使用全轨迹评估（更准确，无跨窗口问题），
        # 超过 80 条时 fallback 到滑动窗口评估（避免超出 LLM context 限制）。
        msg_count = len(simulation.messages) if simulation.messages else 0
        if msg_count <= 80:
            reward_info = TrajectoryEvaluator.calculate_reward_full_traj_rubric(
                task=task,
                full_trajectory=simulation.messages,
                final_state=simulation.states,
                llm_evaluator=llm_evaluator,
                llm_args_evaluator=llm_args_evaluator,
                language=language,
            )
        else:
            reward_info = TrajectoryEvaluator.calculate_reward(
                task=task,
                full_trajectory=simulation.messages,
                final_state=simulation.states,
                llm_evaluator=llm_evaluator,
                llm_args_evaluator=llm_args_evaluator,
                language=language,
            )
    elif evaluation_type == "trajectory_full_traj_rubric":
        # 全轨迹评估（带 rubric）：将完整对话一次性发给 LLM，配合 rubric 列表进行判断。
        reward_info = TrajectoryEvaluator.calculate_reward_full_traj_rubric(
            task=task,
            full_trajectory=simulation.messages,
            final_state=simulation.states,
            llm_evaluator=llm_evaluator,
            llm_args_evaluator=llm_args_evaluator,
            language=language,
        )
    elif evaluation_type == "trajectory_sliding_wo_rubric":
        # 滑动窗口评估（无 rubric）：同样分窗口，但不提供预定义的 rubric 列表，
        # 由 LLM 根据用户指令自行判断任务是否完成。带有 memory 机制在窗口间传递上下文。
        reward_info = TrajectoryEvaluator.calculate_reward_sliding_wo_rubric(
            task=task,
            full_trajectory=simulation.messages,
            final_state=simulation.states,
            llm_evaluator=llm_evaluator,
            llm_args_evaluator=llm_args_evaluator,
            language=language,
        )
    elif evaluation_type == "trajectory_full_traj_wo_rubric":
        # 全轨迹评估（无 rubric）：将完整对话一次性发给 LLM，无 rubric 列表，
        # 由 LLM 综合判断 agent 是否完成了用户指令。
        reward_info = TrajectoryEvaluator.calculate_reward_full_traj_wo_rubric(
            task=task,
            full_trajectory=simulation.messages,
            final_state=simulation.states,
            llm_evaluator=llm_evaluator,
            llm_args_evaluator=llm_args_evaluator,
            language=language,
        )
    elif evaluation_type == "trajectory_full_traj_12dim":
        # 全轨迹评估 + 12 维 capability 框架（不依赖 task 自带 rubric）：
        # LLM 自己识别本场景激活的 capability 维度（A1-C4）+ 逐维判 met，
        # reward = sum(met & activated) / sum(activated)。
        reward_info = TrajectoryEvaluator.calculate_reward_full_traj_12dim(
            task=task,
            full_trajectory=simulation.messages,
            final_state=simulation.states,
            llm_evaluator=llm_evaluator,
            llm_args_evaluator=llm_args_evaluator,
            language=language,
        )
    else:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")
    return reward_info
