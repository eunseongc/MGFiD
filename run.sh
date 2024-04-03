export NGPU=2;
dataset=nq;
if [ ${dataset} = nq ];
then
    text_maxlength=192;
elif [ ${dataset} = tqa ];
then
    text_maxlength=250;
fi
n_contexts=20;
ce_loss_weight=0.5;

sce_loss_weight=1
sce_loss_reduction=sum;
summary_bos_option=add;
sent_summary_pool=max;
sce_loss_fct=focal;
focal_alpha=0.95

seed=0;
ctx_anno=mytho; ## mytho: MathoMax-13B / gpt: ChatGPT / has_answer: whether an answer span is included

name=${dataset}_${n_contexts}_${ctx_anno}${ce_loss_weight}ce$_tf${sce_loss_weight}${sce_loss_fct}sce${focal_alpha}_${summary_bos_option}${sent_summary_pool}_${seed}

echo $name
python -m torch.distributed.launch --nproc_per_node=$NGPU train_reader.py \
    --train_data open_domain_data/${dataset}/train_sent_20_${text_maxlength}_nltk.json \
    --eval_data open_domain_data/${dataset}/dev_sent_20_${text_maxlength}_nltk.json \
    --per_gpu_batch_size 2 \
    --accumulation_steps 16 \
    --n_contexts ${n_contexts} \
    --text_maxlength ${text_maxlength} \
    --ctx_anno ${ctx_anno} \
    --ce_loss_type list \
    --ce_loss_weight ${ce_loss_weight} \
    --passage_pooling first \
    --sce_loss_fct ${sce_loss_fct} \
    --focal_alpha ${focal_alpha} \
    --sce_loss_type point \
    --sce_loss_weight ${sce_loss_weight} \
    --sce_loss_reduction ${sce_loss_reduction} \
    --sentence_pooling mean \
    --seed ${seed} \
    --name ${name} \
    --eval_freq 8000 \
    --total_steps 160000 \
    --use_sent_classifier \
    --use_sent_summary \
    --sent_summary_tf \
    --summary_bos_option ${summary_bos_option} \
    --sent_summary_pool ${sent_summary_pool}

python -m torch.distributed.launch --nproc_per_node=$NGPU test_reader.py --model_path checkpoints/${name}/checkpoint/best_dev --eval_data open_domain_data/${dataset}/test_sent_20_${text_maxlength}_nltk.json
echo $name
