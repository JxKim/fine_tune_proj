"""
单 GPU 训练优化伪代码

这个文件的目标不是“直接运行”，而是把训练中的几个显存优化点拆开讲清楚。

为了方便理解，下面统一使用这些术语：
- sample：一条训练样本
- micro batch：一次真正送进 GPU 的小批次
- global batch / effective batch：累计若干个 micro batch 之后等效的大批次
- step：有时指 dataloader 的一步，有时指 optimizer 更新一步，要注意区分上下文
"""


def training_terms():
    """
    先统一一个最常见的训练配置例子：

    假设：
    - per_device_batch_size = 4
    - gradient_accumulation_steps = 3

    那么：
    - 每次真正进入显存计算的是 4 条样本
    - 连续做 3 次前向 + 反向，但先不更新参数
    - 第 3 次之后才 optimizer.step()
    - 等效 batch size = 4 * 3 = 12

    如果是单卡训练，这个“等效 batch size”就很好理解。
    如果是多卡训练，还要再乘 world_size，但这里我们只讨论单 GPU。
    """

    pass


def pseudo_gradient_accumulation_detail():
    """
    梯度累计细化版。

    为什么需要它：
    - 如果 batch_size=12 放不进显存
    - 但 batch_size=4 能放进去
    - 那就把原来想一次做完的 12 条，拆成 3 次做

    训练时真正发生的事情：
    - 第 1 个 micro batch 反向后，梯度留在参数的 .grad 里
    - 第 2 个 micro batch 反向后，新的梯度继续加到 .grad 里
    - 第 3 个 micro batch 反向后，.grad 里已经是“3 次累计结果”
    - 这时才调用 optimizer.step() 更新参数
    """

    per_device_batch_size = 4
    gradient_accumulation_steps = 3
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps

    global_update_step = 0
    optimizer.zero_grad()

    for dataloader_step, batch in enumerate(train_dataloader):
        # 1. 这里只拿到一个能塞进显存的小 batch
        input_ids = batch["input_ids"].to("cuda")
        labels = batch["labels"].to("cuda")

        # 2. 前向传播
        logits = model(input_ids)
        loss = compute_loss(logits, labels)

        # 3. 关键点：loss 要除以累计步数
        # 目的是让“累计 3 次后的总梯度”与“大 batch 一次性训练”的量级一致
        scaled_loss_for_accum = loss / gradient_accumulation_steps

        # 4. 反向传播
        # 注意：这里不会更新参数，只会把梯度加到 param.grad 上
        scaled_loss_for_accum.backward()

        # 5. 判断当前是否累计够了
        reach_update_boundary = (dataloader_step + 1) % gradient_accumulation_steps == 0

        if reach_update_boundary:
            # 6. 真正更新参数
            optimizer.step()
            lr_scheduler.step()

            # 7. 清梯度，开始下一轮累计
            optimizer.zero_grad()

            global_update_step += 1

            print(
                "完成一次参数更新: ",
                f"global_update_step={global_update_step}, ",
                f"effective_batch_size={effective_batch_size}",
            )

    # 9. 真实代码里还要注意“尾 batch”
    # 如果数据总数不能被 gradient_accumulation_steps 整除，
    # 最后那几步是否也要更新一次，需要额外判断


def pseudo_gradient_accumulation_with_numbers():
    """
    用具体数字再走一遍梯度累计。

    假设：
    - 数据总共 12 条
    - 每个 micro batch 放 4 条
    - gradient_accumulation_steps = 3

    效果上等价于：
    - 1 次参数更新
    - 这次更新看到了 12 条数据
    """

    # 第 1 次
    batch_1 = samples_1_to_4
    loss_1 = forward(batch_1)
    (loss_1 / 3).backward()
    # 当前状态：grad = grad_from_batch_1

    # 第 2 次
    batch_2 = samples_5_to_8
    loss_2 = forward(batch_2)
    (loss_2 / 3).backward()
    # 当前状态：grad = grad_from_batch_1 + grad_from_batch_2

    # 第 3 次
    batch_3 = samples_9_to_12
    loss_3 = forward(batch_3)
    (loss_3 / 3).backward()
    # 当前状态：grad = grad_from_batch_1 + grad_from_batch_2 + grad_from_batch_3

    optimizer.step()
    optimizer.zero_grad()


def pseudo_cpu_offload_detail():
    """
    CPU 卸载细化版。

    先记住训练里最占显存的几类对象：
    - 模型参数 parameters
    - 前向传播产生的激活值 activations
    - 反向传播得到的梯度 gradients
    - 优化器状态 optimizer states

    其中 optimizer states 很容易被忽视，但在 Adam / AdamW 中通常很大：
    - 参数本身一份
    - 一阶动量一份
    - 二阶动量一份
    所以优化器状态可能和参数规模同量级，甚至更夸张。

    CPU 卸载的核心不是“让训练更快”，而是：
    - 把一部分原本待在显存中的东西搬去内存
    - 用 PCIe / NVLink 传输开销换取显存空间
    """

    # 先假设：
    # - 模型主要计算仍在 GPU 上
    # - 优化器状态尽量留在 CPU 上
    device = "cuda"
    model.to(device)

    optimizer = AdamW(model.parameters())
    put_optimizer_state_on_cpu(optimizer)

    for batch in train_dataloader:
        # 1. 当前 batch 必须先进 GPU，因为接下来要前向
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # 2. 前向时：
        # - 当前参与计算的参数在 GPU
        # - 激活值在 GPU
        logits = model(input_ids)
        loss = compute_loss(logits, labels)

        # 3. 反向时：
        # - 梯度在 GPU 上产生
        loss.backward()

        # 4. 更新参数前：
        # 某些框架会把 optimizer state 从 CPU 拉回到“可参与更新的地方”
        # 更新结束后，再把它放回 CPU
        bring_needed_optimizer_state_to_gpu(optimizer)
        optimizer.step()
        send_optimizer_state_back_to_cpu(optimizer)

        # 5. 清理梯度
        optimizer.zero_grad()

        # 6. 有些实现还会把暂时不用的梯度也放回 CPU 或直接释放
        release_unused_gpu_gradients(model)


def pseudo_cpu_offload_memory_flow():
    """
    CPU 卸载最重要的是“对象在哪”。

    可以把一个训练 step 大致想成下面这个流转：

    step 开始前：
    - 参数的一部分在 GPU
    - 优化器状态可能主要在 CPU

    前向传播：
    - batch 在 GPU
    - 当前层参数在 GPU
    - 当前层激活值在 GPU

    反向传播：
    - 梯度在 GPU 上生成

    参数更新：
    - 需要的优化器状态临时参与更新
    - 更新完，能搬回 CPU 的再搬回去

    下一步开始前：
    - GPU 上尽量只保留“下一步马上要用”的东西
    """

    pass


def pseudo_gradient_checkpointing_detail():
    """
    梯度检查点细化版。

    为什么激活值占显存：
    - 前向传播经过很多层
    - 反向传播求梯度时，需要用到这些中间结果
    - 如果每层中间结果都保留，层数越深，显存压力越大

    梯度检查点做了什么：
    - 前向时不保存所有层的中间激活值
    - 只保存少数检查点位置
    - 等到反向传播需要时，再把缺失的那一段重新跑一遍前向

    本质就是：
    - 少存一些
    - 临时多算一些
    """

    hidden_states = token_embedding(input_ids)

    # 假设模型有 4 层，这里把第 1~2 层看成一个 segment，
    # 第 3~4 层看成另一个 segment
    #
    # 不做 checkpoint 时：
    # - layer1, layer2, layer3, layer4 的中间激活都保留在显存里
    #
    # 做 checkpoint 时：
    # - 前向传播时，不保留 segment 内部每一层的完整激活
    # - 只保留这个 segment 的输入 / 输出这类少量信息

    # ---------- forward ----------
    segment_1_input = hidden_states
    hidden_states = checkpoint(segment_1_forward, segment_1_input)

    segment_2_input = hidden_states
    hidden_states = checkpoint(segment_2_forward, segment_2_input)

    logits = lm_head(hidden_states)
    loss = compute_loss(logits, labels)

    # ---------- backward ----------
    # 下面这句是 PyTorch 框架帮我们做的
    loss.backward()

    # 为了辅助理解，你可以把 backward 期间发生的事情脑补成：
    #
    # 1. 反向传播先走到 segment_2
    # 2. 发现：segment_2 内部 layer3 / layer4 的中间激活当初没完整保存
    # 3. 那就用之前保留的 segment_2_input，再重新做一遍前向
    #
    #    recomputed_hidden = segment_2_forward(segment_2_input)
    #
    # 4. 这样就临时拿回了 layer3 / layer4 反向所需的中间结果
    # 5. 然后才能继续计算 segment_2 的梯度
    #
    # 6. 接着反向传播再走到 segment_1
    # 7. 又发现：layer1 / layer2 的中间激活当初也没完整保存
    # 8. 再用保留的 segment_1_input，把 segment_1_forward(segment_1_input) 重算一遍
    # 9. 临时恢复出需要的中间结果后，再继续计算 segment_1 的梯度


def segment_1_forward(hidden_states):
    """
    这一段专门用来说明：
    checkpoint 不是对单层魔法优化，
    而是常常对“一段前向计算”做检查点。
    """

    hidden_states = layer_1(hidden_states)
    hidden_states = layer_2(hidden_states)
    return hidden_states


def segment_2_forward(hidden_states):
    hidden_states = layer_3(hidden_states)
    hidden_states = layer_4(hidden_states)
    return hidden_states


def pseudo_gradient_checkpointing_segment_version():
    """
    另一种更适合理解的写法：把模型按“段”做 checkpoint。

    假设模型有 12 层，把它分成 4 段：
    - segment_1: layer 1~3
    - segment_2: layer 4~6
    - segment_3: layer 7~9
    - segment_4: layer 10~12

    则前向时只在每段边界附近保留少量信息。
    backward 时如果需要 segment_2 中间某层激活值，
    就把 segment_2 重新前向一次算出来。
    """

    hidden_states = token_embedding(input_ids)

    hidden_states = checkpoint(segment_1, hidden_states)
    hidden_states = checkpoint(segment_2, hidden_states)
    hidden_states = checkpoint(segment_3, hidden_states)
    hidden_states = checkpoint(segment_4, hidden_states)

    logits = lm_head(hidden_states)
    loss = compute_loss(logits, labels)
    loss.backward()


def pseudo_mixed_precision_training_detail():
    """
    混合精度训练细化版。

    为什么低精度能省显存：
    - FP32 一般占 4 字节
    - FP16 / BF16 一般占 2 字节
    - 同样数量的张量，低精度通常占用更少显存

    为什么又不能全都无脑低精度：
    - 某些操作对数值范围更敏感
    - 梯度太小可能下溢
    - 某些归一化、累加操作保留 FP32 更稳

    因此实际做法是：
    - 让大部分矩阵乘法、前向计算用低精度
    - 让关键数值操作保留高精度
    """

    use_fp16 = True
    use_bf16 = False

    # FP16 常见地搭配 GradScaler
    scaler = GradScaler(enabled=use_fp16)

    for batch in train_dataloader:
        input_ids = batch["input_ids"].to("cuda")
        labels = batch["labels"].to("cuda")

        # 1. autocast 作用：
        # 自动帮我们决定哪些算子用低精度
        # 你不需要手工把每一层都改成 half()
        with autocast(dtype="float16", enabled=use_fp16):
            logits = model(input_ids)
            loss = compute_loss(logits, labels)

        # 2. FP16 下，为了减少梯度太小直接变成 0 的风险，
        # 常先把 loss 放大，再 backward
        scaler.scale(loss).backward()

        # 3. 如果后面要直接 step，先把缩放关系还原回去
        scaler.unscale_(optimizer)

        # 4. 由 scaler 来执行 step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()


def pseudo_mixed_precision_fp16_vs_bf16():
    """
    FP16 和 BF16 的理解重点：

    FP16：
    - 显存省
    - 算得快
    - 但数值范围和稳定性更敏感
    - 常搭配 GradScaler

    BF16：
    - 也能省显存
    - 指数位更宽，通常更稳定
    - 很多情况下可以不需要 GradScaler

    所以真实训练中常见两种风格：
    - 老一些或通用配置：FP16 + GradScaler
    - 新一些硬件：BF16
    """

    with autocast(dtype="bfloat16"):
        logits = model(input_ids)
        loss = compute_loss(logits, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def pseudo_all_in_one_training_loop_detail():
    """
    这一段把 4 个优化点放进同一个单 GPU 训练循环。

    重点观察每个优化点插入的位置：
    - 梯度检查点：通常在 model 初始化后打开
    - CPU 卸载：通常在 optimizer / 参数管理层面启用
    - 混合精度：包住前向和 loss
    - 梯度累计：控制何时 optimizer.step()
    """

    device = "cuda"
    per_device_batch_size = 2
    gradient_accumulation_steps = 8
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps

    use_gradient_checkpointing = True
    use_cpu_offload = True
    use_amp = True
    amp_dtype = "float16"

    model.to(device)

    # 1. 打开梯度检查点
    # 作用位置：模型内部前向逻辑
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 2. 构建优化器
    # 如果做 CPU 卸载，往往不是普通 AdamW，而是某种带 offload 能力的封装
    if use_cpu_offload:
        optimizer = build_optimizer_with_cpu_offload(model)
    else:
        optimizer = build_optimizer(model)

    lr_scheduler = build_lr_scheduler(optimizer)
    scaler = GradScaler(enabled=use_amp and amp_dtype == "float16")

    model.train()
    optimizer.zero_grad()

    optimizer_update_step = 0

    for dataloader_step, batch in enumerate(train_dataloader):
        # 3. 当前 micro batch 上 GPU
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # 4. 混合精度前向
        with autocast(enabled=use_amp, dtype=amp_dtype):
            logits = model(input_ids)
            loss = compute_loss(logits, labels)

            # 5. 梯度累计要求把 loss 缩小
            loss = loss / gradient_accumulation_steps

        # 6. backward
        # 如果是 FP16，使用 scaler
        if use_amp and amp_dtype == "float16":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # 7. 是否到达一次真正参数更新的边界
        should_step = (dataloader_step + 1) % gradient_accumulation_steps == 0

        if should_step:
            # 8. 更新前常见还会做梯度裁剪
            if use_amp and amp_dtype == "float16":
                scaler.unscale_(optimizer)

            clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 9. 真正更新参数
            if use_amp and amp_dtype == "float16":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # 10. 学习率调度器一般也在“参数更新之后”走一步
            lr_scheduler.step()

            # 11. 清梯度，进入下一轮累计
            optimizer.zero_grad()

            optimizer_update_step += 1
            print(
                "一次参数更新完成: ",
                f"optimizer_update_step={optimizer_update_step}, ",
                f"effective_batch_size={effective_batch_size}",
            )


def pseudo_compare_what_is_saved():
    """
    用一句话对比这 4 个优化主要在“省什么”：

    1. 梯度累计
    - 不是直接减少单次参数总量
    - 而是让你每次只处理更小的 micro batch
    - 从而降低“单次前向/反向”的显存峰值

    2. CPU 卸载
    - 把参数/梯度/优化器状态中的一部分挪到 CPU
    - 直接减少显存常驻内容

    3. 梯度检查点
    - 主要减少激活值保存量
    - 对深层网络很有效

    4. 混合精度训练
    - 通过更低的数据精度，减少张量占用
    - 同时通常还能提高吞吐
    """

    pass


def pseudo_when_to_use_which():
    """
    如果单 GPU 显存不够，通常可以这样理解它们的优先级：

    第一层：
    - 先减小 per_device_batch_size
    - 再配合梯度累计，把等效 batch 撑回来

    第二层：
    - 开混合精度训练
    - 这是最常见、收益也通常很直接的一步

    第三层：
    - 开梯度检查点
    - 适合模型更深、激活值占用明显时

    第四层：
    - 再考虑 CPU 卸载
    - 它更像“用速度换显存”的兜底手段
    """

    pass


def summary_for_understanding():
    """
    最后用一句更准确的话收束：

    梯度累计：
    - 把“大 batch 的更新效果”拆成多个小 batch 来做。

    CPU 卸载：
    - 把不必常驻显存的训练状态临时挪到内存里。

    梯度检查点：
    - 前向少存中间激活，反向再把缺的部分重算出来。

    混合精度训练：
    - 尽量让大多数计算用更低精度完成，从而省显存、提速度。
    """

    pass
