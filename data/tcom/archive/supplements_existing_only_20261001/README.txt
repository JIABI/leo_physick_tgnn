现有 v19 数据的补充表

唯一结果来源仍是本目录上一级 data 中的原文件。本子目录不包含新训练、新仿真、新测试成绩，也没有重新生成原均值或 SD。原 model_id、case_id、seed 和 episode_id 沿用。

新增内容分为两类：
1. 由原记录直接计算的补充量：TGN 测试成本诊断、Greedy 验证成本分项、现有 DQN 测试覆盖与有效数、事件残余计数及可见性必要条件。
2. R1 的拟议规则与参数：只供确认缺失规格，未应用到旧数据。record_type 明确区分 existing_protocol_value、prospective_rule_assumption 和 derived_under_stated_rule_assumption。

TGN 的测试比较必须匹配相同场景。现有 N=2000/H=800 有 Greedy 测试对照；N=500/1000/H=800 没有，不用 Greedy 验证成绩代替测试成绩。
Greedy 验证分项由原有三个 NPZ 复算，保留原有十条验证轨迹及其 J。其原数组路径写在 source_record 中，不复制或改写原数组。
路径约定：source_episode_table 等源表路径相对于本目录上一级 data。沿用的 raw_file 和 SYNTHETIC:09_simulation/... 引用，以 /Users/bella/Downloads/cox-physick/v14_simulated_reference_20260930 为原始根目录；这只是原记录定位，不改变原有来源标记。
实际 selected update 与 selected-checkpoint validation cost 仍缺记录；表中的空值表示尚未恢复，不是零值，也不采用测试数据倒推。

事件表中 attempts−executed_HO−failures 仅表示“尚未细分的尝试”，不能直接改叫物理检查失败数或源连接二次回退数。
实际活跃卫星数只能给可见卫星数的必要下界。缺少原轨道、位置和逐链路记录时，不生成一条新轨迹去充当旧轨迹的证明。
当前 DQN 的机制表整理的是已有声明，不把已有结果改名为文献 successive 算法的复现结果。

上一轮 v19_preexperiment_reference_20261001 中独立生成的测试/验证/物理示例未导入本目录。无需用它替换本轮补充或原 v19 结果。

入口：00_addendum_index.csv。
