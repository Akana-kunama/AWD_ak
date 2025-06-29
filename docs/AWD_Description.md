# AWD

## 文件结构

```
├── awd
│   ├── data
│   │   ├── assets
│   │   │   ├── actuator_nets
│   │   │   ├── go_bdx
│   │   │   │   ├── meshes
│   │   │   │   └── placo_presets
│   │   │   │       └── tmp
│   │   │   ├── mini_bdx
│   │   │   │   ├── meshes
│   │   │   │   └── urdf
│   │   │   └── mjcf
│   │   ├── cfg
│   │   │   ├── go_bdx
│   │   │   │   ├── train
│   │   │   │   └── working_forward walk_only
│   │   │   │       └── train
│   │   │   └── mini_bdx
│   │   │       └── train
│   │   └── motions
│   │       ├── go_bdx
│   │       ├── go_bdx_all
│   │       ├── mini_bdx
│   │       ├── mini_bdx_all_30fps
│   │       └── mini_bdx_all_60fps
│   ├── env
│   │   └── tasks
│   ├── learning
│   ├── poselib
│   │   ├── data
│   │   │   └── configs
│   │   └── poselib
│   │       ├── core
│   │       │   ├── backend
│   │       │   └── tests
│   │       ├── skeleton
│   │       │   └── backend
│   │       │       └── fbx
│   │       └── visualization
│   │           └── tests
│   └── utils
│       └── bdx
├── docs
├── gait_generation
│   └── templates
├── images
└── recordings

```

## 核心脚本

### `run.py`

是整个训练流程的入口，执行顺序如下：

```
main()
 ├─ get_args()                ⬅️ 命令行参数解析
 ├─ load_cfg(args)            ⬅️ 加载 cfg_env, cfg_train YAML 配置
 ├─ set_seed(...)             ⬅️ 设置随机种子
 ├─ create_rlgpu_env(...)     ⬅️ 创建模拟环境（含 Isaac Gym 参数）
 ├─ build_alg_runner(...)     ⬅️ 注册各类算法，构建训练 Runner
 ├─ runner.load(...)          ⬅️ 加载训练配置
 ├─ runner.run(...)           ⬅️ 启动训练 / 推理 / 导出 ONNX

```

**接口/类说明**

+ `main()`

  执行入口

+ `create_rlgpu_env()`

  负责**创建训练环境实例**，并根据配置设定设备、多 GPU 和物理模拟参数。

  关键功能：

  + 检查 `multi_gpu`： 如果使用 Horovod 并行训练，多卡时设置设备号。
  + `parse_sim_params` ： 解析物理仿真器（如 Isaac Gym）的参数，如 substep, solver, buffer size 等
  + `parse_task` : 创建对应的任务环境，如 `DucklingCommand` 返回一个 task 实例 + env 实例
  + `Frame stacking`: 可选地为输入增加时间维度（比如做视觉策略时常见）

  返回的是一个**VecEnv 环境实例**

+ `build_alg_runner(...)`

  注册强化学习算法，并返回一个 `Runner` 实例，用于启动训练或推理。

  注册项：

  | 注册项          | 模块                                                         | 说明                                              |
  | --------------- | ------------------------------------------------------------ | ------------------------------------------------- |
  | algo_factory    | `AMPAgent` / `AWDAgent` / `HRLAgent`                         | 强化学习核心，包含策略更新、判别器更新等          |
  | player_factory  | `AMPPlayerContinuous` / `AWDPlayer` / `HRLPlayer`            | 训练完后进行推理、部署或导出模型的接口            |
  | model_factory   | `ModelAMPContinuous` / `ModelAWDContinuous` / `ModelHRLContinuous` | 把网络打包成可以 forward 和存 checkpoint 的对象   |
  | network_factory | `AMPBuilder` /  `AWDBuilder` / `HRLBuilder`                  | 构建 policy、value、discriminator 网络结构（MLP） |

  > `rl_games` 使用`factory`模式管理算法/模型/网络，只需指定 `algo`，系统就会自动找到 Agent 并构建完整 pipeline

+ `RLGPUAlgoObserver`

  继承自 `rl_games.common.algo_observer.AlgoObserver`，用于记录训练过程中的成功率（如 success 或 consecutive_successes）。

  + `after_init(algo)` 连接到 AMPAgent 或 PPOAgent，准备 logger、记录结构
  + `process_infos()`从环境返回的信息 `infos` 中提取 `successes`（如果开启了 `env.learn.useSuccesses`）
  + `after_clear_stats()` 清空临时统计数据
  + `after_print_stats()` 每轮训练后写入 TensorBoard，如成功率等

  用于**扩展训练过程中的可视化指标**，例如记录成功率、平衡能力、收敛速度等，不影响主流程。

+ `RLGPUEnv`

  封装 `parse_task` 返回的 Isaac Gym 环境，使其**兼容 `rl_games` 的 VecEnv 接口**。

  + `step(action)` 接受一个 batch 的 action，执行环境 step，返回 obs/reward/done/info
  + `reset()` 重置环境，可选择部分重置（用 `env_ids`）
  + `get_env_info()` 提供 obs/action/state 空间的结构，用于模型初始化

  `rl_games` 要求环境是统一的 `IVecEnv` 接口

  Isaac Gym 自带的环境不一定遵守这个接口，所以要用 `RLGPUEnv` 进行封装兼容



### `common_agent.py`

定义了 `CommonAgent` 类，通用 Actor-Critic 强化学习框架，从 `rl_games.algos_torch.a2c_continuous.A2CAgent` 继承的 PPO/Actor-Critic agent，适配 Isaac Gym 多环境并行执行。

- `__init__`
  - 加载策略/值函数网络模型
  - 创建优化器（Adam）
  - 初始化 running mean/std（用于 obs/value 归一化）
  - 初始化 AMP 数据集结构（为了兼容子类）

- `train()`

  训练主循环，执行：

  - 环境 rollout → `play_steps()`
  - 批量训练 policy → `train_epoch()`
  - TensorBoard 记录统计数据 → `log_train_info()`
  - 周期性保存 checkpoint

- #### `train_epoch()`

  训练一轮：

  - 采样经验 `play_steps()`
  - 构造数据集 → `prepare_dataset()`
  - 多轮 `mini_epochs` 更新策略 → `train_actor_critic()`

+ #### `play_steps()`

  收集环境中的一个 rollout：

  ```
  for t in range(horizon_length):
      obs_t → policy → act_t
      act_t → env → obs_{t+1}, reward, done, info
  
  ```

  > **rollout** 是按照当前策略 π(a|s) 在环境中进行一次或多步交互的过程，记录：
  > $$
  > {(s_0​,a_0​,r_0​),(s_1​,a_1​,r_1​),…,(s_T​,a_T​,r_T​)}
  > \\
  > 其中：\\
  > 
  >     s_{t}​：第 t 步的状态（observation）\\
  > 
  >     a_{t}​：策略选择的动作\\
  > 
  >     r_{t}​：执行动作后环境返回的奖励\\
  > 
  >     T：rollout 的最大步长（通常固定，如 32 步、128 步）\\
  > $$
  > 常见的 rollout buffer 会包含以下张量：
  >
  > | 名称        | 说明                         |
  > | ----------- | ---------------------------- |
  > | `obses`     | 状态序列                     |
  > | `actions`   | 动作序列                     |
  > | `rewards`   | 奖励序列                     |
  > | `dones`     | 是否结束                     |
  > | `values`    | critic 网络的状态值          |
  > | `log_probs` | 策略网络的 log π(a           |
  > | `amp_obs`   | 如果是 AMP，还会存判别器输入 |

  并记录：

  - `obses`, `rewards`, `dones`, `values`, `actions`
  - 计算 advantage + return（GAE）

### `amp_agent.py`

定义了 `AMPAgent` 类，是 `CommonAgent` 的子类，**扩展支持 AMP（Adversarial Motion Priors）模仿学习**

> ### AMP 模型核心思想
>
> AMP 通过训练一个判别器 `D(obs)` 判别动作是否源自参考动作或 agent 的 rollouts，从而生成“模仿奖励”。
>
> 最终 agent 的 total reward 变成：
> $$
> r=w_{task}​⋅r_{env}​+w_{disc}​⋅r_{amp} \\
> $$
>
> | 符号   | 含义                         | 数据来源                                    | 控制因素                                      |
> | ------ | ---------------------------- | ------------------------------------------- | --------------------------------------------- |
> | r      | **最终奖励**                 | 用于 PPO 策略梯度更新                       | -                                             |
> | r_envr | **环境奖励（task reward）**  | 由环境返回，如速度、到达目标、保持平衡等    | `env.learn.*` 中定义的 reward scale           |
> | r_amp  | **判别器奖励（AMP reward）** | 判别器越分不清“假动作”和“真动作” → 奖励越高 | 由 `AMPAgent._calc_disc_rewards()` 计算       |
> | w_task | **任务奖励权重**             | 控制 r_env 在总奖励中的占比                 | 由 `task_reward_w` 参数控制（通常在 YAML 中） |
> | w_disc | **模仿奖励权重**             | 控制 r_amp 在总奖励中的占比                 | 由 `disc_reward_w` 参数控制                   |

+ 重要成员变量和函数：

  + `self._amp_obs_demo_buffer` 存储 ground-truth demo AMP 观察
  + `self._amp_replay_buffer` 存储历史 rollout 的 AMP 观察
  + `self._eval_disc()` 用当前策略计算判别器 score
  + `_calc_disc_rewards()` 将判别器 logit 转换为 reward
  + `_disc_loss()` 判别器 BCE 损失 + 正则项
  + `_combine_rewards()` 混合任务奖励 + 判别器奖励
  + `_disc_loss() ` 判别器 BCE 损失 + 正则项
  + `_combine_rewards()` 混合任务奖励 + 判别器奖励

+ `init_tensors()`：

  扩展经验缓存，添加：

  - `amp_obs`（判别器观察）

  - `rand_action_mask`（用于 ε-greedy 模仿）

- 重载主逻辑

  - #### `train_epoch()`：

    - 正常 rollout 之后，还会：

      - 从 demo buffer 采样 `amp_obs_demo`
      - 从 replay buffer 采样 `amp_obs_replay`
      - 组合判别器训练数据

    - 最终计算 AMP loss，并整合进总 loss：

      ```
      loss = actor + critic - entropy + bounds_loss + disc_loss
      ```

  + `play_steps()`

    + 记录 `amp_obs`, `rand_action_mask`

    + 计算 `disc_reward = -log(1 - D(obs))`

    + 和 task_reward 组合为 `combined_reward`

      ```python
      combined_rewards = task_reward_w * task_reward + disc_reward_w * disc_reward
      ```

### `hrl_agent.py`

HRLAgent类是一个 **分层强化学习（Hierarchical RL）代理**，其中高层策略（HLC）输出 latent vector（潜变量），底层策略（LLC）执行低层动作。这种设计支持高层目标调度与底层精细控制的分离。