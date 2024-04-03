export NGPU=2

path=$1

if [[ $path == *"nq"* ]]; then
    eval_data=open_domain_data/nq/test_sent_20_192_nltk.json
elif [[ $path == *"tqa"* ]]; then
    eval_data=open_domain_data/tqa/test_sent_20_250_nltk.json
fi

python -m torch.distributed.launch --nproc_per_node=$NGPU test_reader.py --model_path ${path} \
                                                                         --eval_data ${eval_data} \
                                                                         --per_gpu_batch_size 1
