#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
REGION="${REGION:-europe-west4}"
ZONE="${ZONE:-europe-west4-a}"
VM="${VM:-eval-vlm-l4}"
NETWORK="${NETWORK:-eval-vpc}"
SUBNET="${SUBNET:-eval-subnet}"

gcloud config set project "$PROJECT_ID"
gcloud compute instances delete "$VM" --zone="$ZONE" --delete-disks=all --quiet
gcloud compute routers nats delete eval-nat --router=eval-router --region="$REGION" --quiet || true
gcloud compute routers delete eval-router --region="$REGION" --quiet || true
gcloud compute firewall-rules delete allow-iap-ssh --quiet || true
gcloud compute networks subnets delete "$SUBNET" --region="$REGION" --quiet || true
gcloud compute networks delete "$NETWORK" --quiet || true
