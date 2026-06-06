#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-europe-west4}"
ZONE="${ZONE:-europe-west4-a}"
VM="${VM:-eval-vlm-l4}"
NETWORK="${NETWORK:-eval-vpc}"
SUBNET="${SUBNET:-eval-subnet}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-nvidia-l4}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
DISK_SIZE="${DISK_SIZE:-200GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
KEYRING="${KEYRING:-eval-keyring}"
KEY="${KEY:-eval-disk-key}"
SA="${SA:-eval-vlm-vm}"
CIDR="${CIDR:-10.10.0.0/24}"

gcloud config set project "$PROJECT_ID"
gcloud services enable compute.googleapis.com iap.googleapis.com cloudkms.googleapis.com

gcloud compute networks describe "$NETWORK" >/dev/null 2>&1 || gcloud compute networks create "$NETWORK" --subnet-mode=custom
gcloud compute networks subnets describe "$SUBNET" --region="$REGION" >/dev/null 2>&1 || gcloud compute networks subnets create "$SUBNET" --network="$NETWORK" --region="$REGION" --range="$CIDR" --enable-private-ip-google-access
gcloud compute firewall-rules describe allow-iap-ssh >/dev/null 2>&1 || gcloud compute firewall-rules create allow-iap-ssh --network="$NETWORK" --allow=tcp:22 --source-ranges=35.235.240.0/20
gcloud compute routers describe eval-router --region="$REGION" >/dev/null 2>&1 || gcloud compute routers create eval-router --network="$NETWORK" --region="$REGION"
gcloud compute routers nats describe eval-nat --router=eval-router --region="$REGION" >/dev/null 2>&1 || gcloud compute routers nats create eval-nat --router=eval-router --region="$REGION" --nat-all-subnet-ip-ranges --auto-allocate-nat-external-ips

gcloud compute project-info add-metadata --metadata enable-oslogin=TRUE
gcloud iam service-accounts describe "$SA@$PROJECT_ID.iam.gserviceaccount.com" >/dev/null 2>&1 || gcloud iam service-accounts create "$SA" --display-name="Eval VLM VM"
gcloud kms keyrings describe "$KEYRING" --location="$REGION" >/dev/null 2>&1 || gcloud kms keyrings create "$KEYRING" --location="$REGION"
gcloud kms keys describe "$KEY" --location="$REGION" --keyring="$KEYRING" >/dev/null 2>&1 || gcloud kms keys create "$KEY" --location="$REGION" --keyring="$KEYRING" --purpose=encryption

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
gcloud kms keys add-iam-policy-binding "$KEY" --location="$REGION" --keyring="$KEYRING" --member="serviceAccount:service-${PROJECT_NUMBER}@compute-system.iam.gserviceaccount.com" --role="roles/cloudkms.cryptoKeyEncrypterDecrypter" >/dev/null

gcloud compute instances describe "$VM" --zone="$ZONE" >/dev/null 2>&1 || gcloud compute instances create "$VM" --zone="$ZONE" --machine-type="$MACHINE_TYPE" --accelerator="type=$ACCELERATOR_TYPE,count=$ACCELERATOR_COUNT" --maintenance-policy=TERMINATE --provisioning-model=SPOT --instance-termination-action=STOP --image-family="$IMAGE_FAMILY" --image-project=deeplearning-platform-release --boot-disk-size="$DISK_SIZE" --boot-disk-type=pd-ssd --boot-disk-kms-key="projects/$PROJECT_ID/locations/$REGION/keyRings/$KEYRING/cryptoKeys/$KEY" --network="$NETWORK" --subnet="$SUBNET" --no-address --service-account="$SA@$PROJECT_ID.iam.gserviceaccount.com" --no-scopes --shielded-vtpm --shielded-integrity-monitoring --metadata=block-project-ssh-keys=TRUE,enable-oslogin=TRUE,serial-port-enable=FALSE

echo "VM ready: $VM in $ZONE"
echo "SSH: gcloud compute ssh $VM --zone=$ZONE --tunnel-through-iap"
