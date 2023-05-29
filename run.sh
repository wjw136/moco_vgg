time=$(date "+%Y%m%d-%H%M%S")



max_len=4
moco_k=320
fintune_lr=1e-4
model_path="/home/server8/jwwang/Data/model/vggish-10086976.pth"
device=0
seed=10
fintune_epoch=20
use_attention=false
d
NUM_FRAMES=99
NUM_BANDS=50



echo "pretrain stage-moco"
pretrain_data_path2="/home/server8/jwwang/Data/ContrastData"
pretrain_model_path2="/home/server8/jwwang/Data/pretrainModel/vgg_contrastPretrain_${NUM_FRAMES}x${NUM_BANDS}.pt"
sed -i s/"^NUM_FRAMES.*$"/"NUM_FRAMES = ${NUM_FRAMES}"/ '/home/server8/jwwang/moco_SA/modelBuilder/torchvggish/vggish_params.py'
sed -i s/"^NUM_BANDS.*$"/"NUM_BANDS = ${NUM_BANDS}"/ '/home/server8/jwwang/moco_SA/modelBuilder/torchvggish/vggish_params.py'
python -u main_moco.py \
    --batch_size 32 \
    --epoch 15 \
    --learning_rate "$fintune_lr" \
    --data_dir "$pretrain_data_path2" \
    --save_path "$pretrain_model_path2" \
    --augment True \
    --is_pretrained True \
    --pretrained_path "$model_path" \
    --model_name_or_path "$model_path" \
    --moco_k "$moco_k" \
    --cuda 0 \
    --seed "$seed" \
    --use_attention ${use_attention} \
    --max_len ${max_len}



echo fintune multi model
for i in $(seq 0 6); do
    echo "***************************************************************************************************"
    echo "train stage-model${i}"
    train_lr=1e-4
    train_data_dir="/home/server8/jwwang/Data/ExpData"
    save_path="/home/server8/jwwang/Data/finalModel/vgg4clf_${time}_model${i}.pt"
    train_name="down-train-${i}"
    mode=train
    sed -i s/"^NUM_FRAMES.*$"/"NUM_FRAMES = ${NUM_FRAMES}"/ '/home/server8/jwwang/moco_SA/modelBuilder/torchvggish/vggish_params.py'
    sed -i s/"^NUM_BANDS.*$"/"NUM_BANDS = ${NUM_BANDS}"/ '/home/server8/jwwang/moco_SA/modelBuilder/torchvggish/vggish_params.py'
    python -u main_lincls.py \
        --seed "$seed" \
        --cuda 0 \
        --num_train_epochs "${fintune_epoch}" \
        --learning_rate "$train_lr" \
        --data_dir "$train_data_dir" \
        --save_path "$save_path" \
        --model_name_or_path "$model_path" \
        --is_pretrained true \
        --pretrained_path "$pretrain_model_path2" \
        --train_name "$train_name" \
        --mode "$mode" \
        --use_attention "$use_attention"
done


echo "max-vote***************************"
train_data_dir="/home/server8/jwwang/Data/ExpData"
mode=test
sed -i s/"^NUM_FRAMES.*$"/"NUM_FRAMES = ${NUM_FRAMES}"/ '/home/server8/jwwang/moco_SA/modelBuilder/torchvggish/vggish_params.py'
sed -i s/"^NUM_BANDS.*$"/"NUM_BANDS = ${NUM_BANDS}"/ '/home/server8/jwwang/moco_SA/modelBuilder/torchvggish/vggish_params.py'
python -u main_lincls.py \
    --seed "$seed" \
    --cuda 0 \
    --data_dir "$train_data_dir" \
    --model_name_or_path "$model_path" \
    --model_list "/home/server8/jwwang/Data/finalModel/vgg4clf_${time}_model0.pt, /home/server8/jwwang/Data/finalModel/vgg4clf_${time}_model1.pt,
    /home/server8/jwwang/Data/finalModel/vgg4clf_${time}_model2.pt, /home/server8/jwwang/Data/finalModel/vgg4clf_${time}_model3.pt, /home/server8/jwwang/Data/finalModel/vgg4clf_${time}_model4.pt,
    /home/server8/jwwang/Data/finalModel/vgg4clf_${time}_model5.pt, /home/server8/jwwang/Data/finalModel/vgg4clf_${time}_model6.pt" \
    --mode "$mode" \
    --use_attention "$use_attention"

