if [ "$RUN_TYPE" == "train" ]
then
    accelerate launch --multi_gpu /home/workspace/treebeard_vit/train.py --config_file /home/workspace/config/accelerator.yaml
elif [ "$RUN_TYPE" == "test" ]
then
    accelerate launch /home/workspace/treebeard_vit/test.py --config_file /home/workspace/config/accelerator.yaml
elif [ "$RUN_TYPE" == "api" ]
then
    echo "Not ready"
else
    echo "Run type Error, please set environment variable properly"
fi

