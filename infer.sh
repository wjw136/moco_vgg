

NUM_FRAMES=99
NUM_BANDS=50

max_len=4
moco_k=320
fintune_lr=1e-4
model_path="/home/server8/jwwang/Data/model/vggish-10086976.pth"
device=0
seed=10

time="20230526-145447"
use_attention=false

echo "max-vote--[INFER]***************************"
train_data_dir="/home/server8/jwwang/Data/ExpData"
mode=test
sed -i s/"^NUM_FRAMES.*$"/"NUM_FRAMES = ${NUM_FRAMES}"/ '/home/server8/jwwang/moco_SA/modelBuilder/torchvggish/vggish_params.py'
sed -i s/"^NUM_BANDS.*$"/"NUM_BANDS = ${NUM_BANDS}"/ '/home/server8/jwwang/moco_SA/modelBuilder/torchvggish/vggish_params.py'
python -u main_lincls_plot.py \
    --seed "$seed" \
    --cuda 0 \
    --data_dir "$train_data_dir" \
    --model_name_or_path "$model_path" \
    --model_list "/home/server8/jwwang/Data/finalModel/vgg4clf_${time}_model0.pt, /home/server8/jwwang/Data/finalModel/vgg4clf_${time}_model1.pt,
    /home/server8/jwwang/Data/finalModel/vgg4clf_${time}_model2.pt, /home/server8/jwwang/Data/finalModel/vgg4clf_${time}_model3.pt,
    /home/server8/jwwang/Data/finalModel/vgg4clf_${time}_model4.pt, /home/server8/jwwang/Data/finalModel/vgg4clf_${time}_model5.pt,
    /home/server8/jwwang/Data/finalModel/vgg4clf_${time}_model6.pt" \
    --mode "$mode" \
    --use_attention "$use_attention"