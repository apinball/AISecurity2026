This folder is to save all records of experiments.

The folder structure is as follows: 
```
eg.
record/
    attack1/
        bd_test_dataset/
        bd_train_dataset/
        defense/ (all defense results following this attack)
            abl/
            ac/
            ...
        xxxx.log
        attack_df.csv
        attack_df_summary.csv
        attack_result.pt (attack result pt file used in following defenses)
        ...
    attack2/
    ...
```

## 팀원 결과 통합본 (heavy artifacts, git 미포함)

- `lf_team/`, `sig_team/`, `badnet_team/`, `blended_team/` … 조수빈(LF), SIG, BadNets, Blended 담당자 결과 통합본
  - `badnet_team/checkpoints/`: BadNets 공격 + 6방어(ABL/FP/FT/NAD/ANP/NC) `.pt` (`team/badnet/README.md` 참조)
  - `blended_team/checkpoints/`: Blended 공격 + 5방어(ABL/ANP/FT/NAD/NC) `.pt` (`team/blended/README.md` 참조)