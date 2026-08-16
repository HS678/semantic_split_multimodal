# Codex Instructions

## Communication

- 所有解释、计划、问题、总结使用中文。
- 命令、代码、路径、配置字段、错误信息保持原文。
- 修改任何文件前必须说明：修改文件、修改原因、影响范围。
- 不确定时先问；不自行改变实验设计；不做无关大规模重构。

## Coding And Validation

- 优先最小修改，保持现有接口和结构。
- 不引入未经验证的新依赖。
- 新增模块必须说明位置、职责、与现有代码关系。
- 修改代码后至少执行 compile/import check；涉及流程时执行相关 pytest 或 smoke run。
- 删除文件前确认用途、引用和替代版本。

## Git

禁止自动执行：

```text
git commit
git push
git reset
git checkout
```

允许查看：

```text
git status
git diff
git log
```

## Completion Report

完成后用中文总结：

- 新增 / 修改 / 删除文件
- 修改内容与原因
- 是否影响算法协议和实验结果
- Validation 命令与结果
- Risks
- 如有 tracked 修改，给出建议 commit message，不自动 commit
