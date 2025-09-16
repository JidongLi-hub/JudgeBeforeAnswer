# Judge Before Answer


## 评测方法
- 评测哪个模型就用Judge_before_Answer/test_model.sh中的相应命令在一个终端中启动该模型
- 接着用Judge_before_Answer/test.py 生成受测试模型的实验结果
- 再用Judge_before_Answer/evaluate.py 计算评测指标

## Updates
#### 12/9/25
- 评测集生成pipline完成
- prompt生成类完成
- 实体前提错误生成完成

## To Do 
- [ ] 完成其他各种类型前提的pormpt生成
- [ ] GRPO训练
- [ ] 模型评测