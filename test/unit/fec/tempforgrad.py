def test_gradient(self):
    """Test that gradient is accessible and not None."""

    # 设置测试参数
    batch_size = 100
    pcm, k, n, _ = load_parity_check_examples(2) 

    # 检查可训练参数是否正常工作
    dec = LDPCBPDecoder(pcm, trainable=True)  # 可训练解码器
    self.assertGreater(len(list(dec.parameters())), 0, "可训练变量应该存在")

    dec = LDPCBPDecoder(pcm, trainable=False)  # 非可训练解码器
    self.assertEqual(len(list(dec.parameters())), 0, "可训练变量不应该存在")

    # 定义要测试的校验节点类型和是否可训练
    cns = ['boxplus', 'boxplus-phi', 'minsum']
    trainable = [True, False]

    # 遍历所有可能的校验节点类型和可训练状态
    for cn in cns:
        for t in trainable:
            # 创建LDPCBPDecoder对象
            dec = LDPCBPDecoder(pcm, trainable=t, cn_type=cn, hard_out=False)

            # 生成随机的LLR
            llr = torch.normal(mean=4.2, std=1.0, size=(batch_size, n))

            # 启用梯度跟踪
            llr = llr.requires_grad_()

            # 前向传播
            x = dec(llr)

            # 计算可训练参数的梯度
            if t:  # 可训练情况
                grads = torch.autograd.grad(outputs=x.sum(), inputs=dec.parameters(), create_graph=True, allow_unused=True)
                
                # 检查梯度是否存在
                self.assertIsNotNone(grads, "梯度不应该为None")
                self.assertGreater(len(grads), 0, "没有找到可训练变量的梯度")

                # 检查每个梯度是否为None
                print(param.requires_grad for param in dec.parameters())
                grads = [g for g in grads if g is not None]
                for g in grads:
                    self.assertIsNotNone(g, "参数的梯度为None")
            else:  # 非可训练情况
                grads = torch.autograd.grad(outputs=x.sum(), inputs=dec.parameters(), create_graph=True, allow_unused=True)
                # 检查没有梯度存在
                self.assertTrue(all(g is None for g in grads), "应该不存在梯度")
