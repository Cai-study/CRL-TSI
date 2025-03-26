#!/bin/bash

while getopts 'e:c:i:l:w:t:n:d:' OPT; do
    case $OPT in
        e) exp=$OPTARG;;
        c) cuda=$OPTARG;;
		    i) identifier=$OPTARG;;
		    l) lr=$OPTARG;;
		    w) stu_w=$OPTARG;;
		    t) task=$OPTARG;;
		    n) num_epochs=$OPTARG;;
		    d) train_flag=$OPTARG;;
    esac
done
echo "exp:" $exp
echo "cuda:" $cuda
echo "num_epochs:" $num_epochs
echo "train_flag:" $train_flag





if [ ${task} = "mmwhs_ct2mr" ];
  then
    labeled_data="train_ct2mr_labeled"
    unlabeled_data="train_ct2mr_unlabeled"
    eval_data="eval_ct2mr"
    test_data="test_mr"
    modality="MR"
    folder="Exp_UDA_MMWHS_ct2mr/"

    if [ ${train_flag} = "true" ]; then
      python CRL-TSI/train_${exp}.py --exp ${folder}${exp}${identifier}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -w ${stu_w} -ep ${num_epochs} -sl ${labeled_data} -su ${unlabeled_data} -se ${eval_data} -t ${task}
    fi
    python CRL-TSI/test.py --exp ${folder}${exp}${identifier}/fold1 -g ${cuda} --split ${test_data} -t ${task}
    python CRL-TSI/evaluate.py --exp ${folder}${exp}${identifier} --folds 1  --split ${test_data} --modality ${modality} -t ${task}
fi


if [ ${task} = "mmwhs_mr2ct" ];
  then
    labeled_data="train_mr2ct_labeled"
    unlabeled_data="train_mr2ct_unlabeled"
    eval_data="eval_mr2ct"
    test_data="test_ct"
    modality="CT"
    folder="Exp_UDA_MMWHS_mr2ct/"
    if [ ${train_flag} = "true" ]; then
      python CRL-TSI/train_${exp}.py --exp ${folder}${exp}${identifier}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -w ${stu_w} -ep ${num_epochs} -sl ${labeled_data} -su ${unlabeled_data} -se ${eval_data} -t ${task}
    fi
    python CRL-TSI/test.py --exp ${folder}${exp}${identifier}/fold1 -g ${cuda} --split ${test_data} -t ${task}
    python CRL-TSI/evaluate.py --exp ${folder}${exp}${identifier} --folds 1 --split ${test_data} --modality ${modality} -t ${task}
fi


if [ ${task} = "abdominal_ct2mr" ];
  then
    labeled_data="train_ct2mr_labeled"
    unlabeled_data="train_ct2mr_unlabeled"
    eval_data="eval_ct2mr"
    test_data="test_mr"
    modality="MR"
    folder="Exp_UDA_abdominal_ct2mr/"

    if [ ${train_flag} = "true" ]; then
      python CRL-TSI/train_${exp}.py --exp ${folder}${exp}${identifier}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -w ${stu_w} -ep ${num_epochs} -sl ${labeled_data} -su ${unlabeled_data} -se ${eval_data} -t ${task}
    fi
    python CRL-TSI/test.py --exp ${folder}${exp}${identifier}/fold1 -g ${cuda} --split ${test_data} -t ${task}
    python CRL-TSI/evaluate.py --exp ${folder}${exp}${identifier} --folds 1  --split ${test_data} --modality ${modality} -t ${task}
fi


if [ ${task} = "abdominal_mr2ct" ];
  then
    labeled_data="train_mr2ct_labeled"
    unlabeled_data="train_mr2ct_unlabeled"
    eval_data="eval_mr2ct"
    test_data="test_ct"
    modality="CT"
    folder="Exp_UDA_abdominal_mr2ct/"
    if [ ${train_flag} = "true" ]; then
      python CRL-TSI/train_${exp}.py --exp ${folder}${exp}${identifier}/fold1 --seed 0 -g ${cuda} --base_lr ${lr} -w ${stu_w} -ep ${num_epochs} -sl ${labeled_data} -su ${unlabeled_data} -se ${eval_data} -t ${task}
    fi
    python CRL-TSI/test.py --exp ${folder}${exp}${identifier}/fold1 -g ${cuda} --split ${test_data} -t ${task}


    python CRL-TSI/evaluate.py --exp ${folder}${exp}${identifier} --folds 1 --split ${test_data} --modality ${modality} -t ${task}
fi
