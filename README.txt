Readme


1. System requirements
All software dependencies and operating systems (including version numbers)


* torch-2.0.1
* torchvision-0.15.2
* beartype-0.17.1
* loguru-0.7.2
* opencv-python-4.10.0.84 
* plotly-5.24.1 
* scikit-image-0.21.0
* volumentations-3D-1.0.4
* pydicom-2.4.2
* pylibjpeg-1.4.0
* pylibjpeg-libjpeg-1.3.4
* python-gdcm-3.0.22


Versions the software has been tested on
* Python 3.8


Any required non-standard hardware
* Nvidia PCIe A100 GPU (80GB)
* Running on AzureML Standard_NC24ads_A100_v4 (24 cores, 220 GB RAM, 64 GB disk)
* Nvidia Tesla V100 GPU (32GB)
* Running on AzureML Standard_NC6s_v3 (6 cores, 112 GB RAM, 736 GB disk)


2. Installation guide
Instructions
* pip install beartype==0.17.1 einops==0.7.0 volumentations-3D beartype==0.17.1 loguru==0.7.2
* Other packages are pre-built by AzureML docker images:
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu117-py38-torch201:biweekly.202312.2


# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir


# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN /var/requirements/install_system_requirements.sh && \
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888


# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update
RUN apt-get install -y openssh-server openssh-client


Typical install time on a "normal" desktop computer
* Less than 10 secs for pip install
* 10 minutes of docker images


3. Demo
Instructions to run on data
* python main.py –output_dir <output_dir> –batch_size 8 –num_workers 6 –lr 1e-5 –wd 0 –num_epochs 20 – lr_decay 15 –train_val_dir <input_train_val dir> –seed 42 –weight_dir <GenerateCT pretrain dir>


Expected output
* Csv file with training/validation/testing samples prediction
* Model checkpoint after training


Expected run time for demo on a "normal" desktop computer
* <what we claim in the paper>


4. Instructions for use
How to run the software on your data
* Preprocessed input images to shape (164,164,164) [A demo of preprocessing is provided using open source Chest CT images (Link, Paper).]
   * Link: https://github.com/hasibzunair/3D-image-classification-tutorial?tab=readme-ov-file 
   * Paper: https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1 
* Adjust batch size, learning rate
* Adjust input/output direction
* Run training command