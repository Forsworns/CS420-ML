# lenet实现在了/opt/caffe下
docker run -it bvlc/caffe:cpu_kyrie

# 用jupyter打开
# docker run -it -p 8888:8888 bvlc/caffe:cpu_kyrie sh -c "jupyter notebook --no-browser --ip 0.0.0.0 /opt/caffe/ --allow-root"