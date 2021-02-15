docker run \

-p 5200:8501 \

--mount type=bind,source=/home/liuziyu004/self_project/docker_make/models/add_sub,target=/models/add_sub1 \

-e MODEL_NAME=add_sub \

-t tensorflow/serving &
