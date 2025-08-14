# Terraform configuration for Photonic Flash Attention infrastructure

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

# AWS Provider configuration
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = "photonic-flash-attention"
      ManagedBy   = "terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC for the cluster
resource "aws_vpc" "photonic_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "photonic-vpc-${var.environment}"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "photonic_igw" {
  vpc_id = aws_vpc.photonic_vpc.id

  tags = {
    Name = "photonic-igw-${var.environment}"
  }
}

# Subnets
resource "aws_subnet" "photonic_public_subnets" {
  count = min(length(data.aws_availability_zones.available.names), 3)

  vpc_id                  = aws_vpc.photonic_vpc.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "photonic-public-subnet-${count.index + 1}-${var.environment}"
    Type = "public"
  }
}

resource "aws_subnet" "photonic_private_subnets" {
  count = min(length(data.aws_availability_zones.available.names), 3)

  vpc_id            = aws_vpc.photonic_vpc.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "photonic-private-subnet-${count.index + 1}-${var.environment}"
    Type = "private"
  }
}

# Route Tables
resource "aws_route_table" "photonic_public_rt" {
  vpc_id = aws_vpc.photonic_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.photonic_igw.id
  }

  tags = {
    Name = "photonic-public-rt-${var.environment}"
  }
}

resource "aws_route_table_association" "photonic_public_rta" {
  count = length(aws_subnet.photonic_public_subnets)

  subnet_id      = aws_subnet.photonic_public_subnets[count.index].id
  route_table_id = aws_route_table.photonic_public_rt.id
}

# NAT Gateways for private subnets
resource "aws_eip" "photonic_nat_eips" {
  count = var.enable_nat_gateway ? min(length(aws_subnet.photonic_public_subnets), var.nat_gateway_count) : 0

  domain = "vpc"
  depends_on = [aws_internet_gateway.photonic_igw]

  tags = {
    Name = "photonic-nat-eip-${count.index + 1}-${var.environment}"
  }
}

resource "aws_nat_gateway" "photonic_nat_gw" {
  count = var.enable_nat_gateway ? min(length(aws_subnet.photonic_public_subnets), var.nat_gateway_count) : 0

  allocation_id = aws_eip.photonic_nat_eips[count.index].id
  subnet_id     = aws_subnet.photonic_public_subnets[count.index].id

  tags = {
    Name = "photonic-nat-gw-${count.index + 1}-${var.environment}"
  }

  depends_on = [aws_internet_gateway.photonic_igw]
}

resource "aws_route_table" "photonic_private_rt" {
  count = var.enable_nat_gateway ? length(aws_subnet.photonic_private_subnets) : 0

  vpc_id = aws_vpc.photonic_vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.photonic_nat_gw[count.index % length(aws_nat_gateway.photonic_nat_gw)].id
  }

  tags = {
    Name = "photonic-private-rt-${count.index + 1}-${var.environment}"
  }
}

resource "aws_route_table_association" "photonic_private_rta" {
  count = var.enable_nat_gateway ? length(aws_subnet.photonic_private_subnets) : 0

  subnet_id      = aws_subnet.photonic_private_subnets[count.index].id
  route_table_id = aws_route_table.photonic_private_rt[count.index].id
}

# Security Groups
resource "aws_security_group" "photonic_cluster_sg" {
  name_prefix = "photonic-cluster-sg-${var.environment}-"
  vpc_id      = aws_vpc.photonic_vpc.id

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Cluster communication"
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    self        = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "photonic-cluster-sg-${var.environment}"
  }
}

# EKS Cluster IAM Role
resource "aws_iam_role" "photonic_cluster_role" {
  name = "photonic-cluster-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "photonic_cluster_AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.photonic_cluster_role.name
}

# EKS Node Group IAM Role
resource "aws_iam_role" "photonic_node_role" {
  name = "photonic-node-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "photonic_node_AmazonEKSWorkerNodePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.photonic_node_role.name
}

resource "aws_iam_role_policy_attachment" "photonic_node_AmazonEKS_CNI_Policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.photonic_node_role.name
}

resource "aws_iam_role_policy_attachment" "photonic_node_AmazonEC2ContainerRegistryReadOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.photonic_node_role.name
}

# EKS Cluster
resource "aws_eks_cluster" "photonic_cluster" {
  name     = "photonic-cluster-${var.environment}"
  role_arn = aws_iam_role.photonic_cluster_role.arn
  version  = var.kubernetes_version

  vpc_config {
    subnet_ids              = concat(aws_subnet.photonic_public_subnets[*].id, aws_subnet.photonic_private_subnets[*].id)
    security_group_ids      = [aws_security_group.photonic_cluster_sg.id]
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = var.cluster_endpoint_public_access_cidrs
  }

  encryption_config {
    provider {
      key_arn = aws_kms_key.photonic_cluster_key.arn
    }
    resources = ["secrets"]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.photonic_cluster_AmazonEKSClusterPolicy,
    aws_cloudwatch_log_group.photonic_cluster_logs,
  ]

  tags = {
    Name = "photonic-cluster-${var.environment}"
  }
}

# CloudWatch Log Group for EKS
resource "aws_cloudwatch_log_group" "photonic_cluster_logs" {
  name              = "/aws/eks/photonic-cluster-${var.environment}/cluster"
  retention_in_days = var.cloudwatch_log_retention_days

  tags = {
    Name = "photonic-cluster-logs-${var.environment}"
  }
}

# KMS Key for EKS encryption
resource "aws_kms_key" "photonic_cluster_key" {
  description             = "KMS key for Photonic EKS cluster encryption"
  deletion_window_in_days = 7

  tags = {
    Name = "photonic-cluster-key-${var.environment}"
  }
}

resource "aws_kms_alias" "photonic_cluster_key_alias" {
  name          = "alias/photonic-cluster-${var.environment}"
  target_key_id = aws_kms_key.photonic_cluster_key.key_id
}

# EKS Node Groups
resource "aws_eks_node_group" "photonic_general_nodes" {
  cluster_name    = aws_eks_cluster.photonic_cluster.name
  node_group_name = "photonic-general-${var.environment}"
  node_role_arn   = aws_iam_role.photonic_node_role.arn
  subnet_ids      = aws_subnet.photonic_private_subnets[*].id

  capacity_type  = "ON_DEMAND"
  instance_types = var.general_node_instance_types

  scaling_config {
    desired_size = var.general_node_desired_size
    max_size     = var.general_node_max_size
    min_size     = var.general_node_min_size
  }

  update_config {
    max_unavailable = 1
  }

  labels = {
    role = "general"
  }

  depends_on = [
    aws_iam_role_policy_attachment.photonic_node_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.photonic_node_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.photonic_node_AmazonEC2ContainerRegistryReadOnly,
  ]

  tags = {
    Name = "photonic-general-nodes-${var.environment}"
  }
}

# GPU Node Group (for photonic acceleration)
resource "aws_eks_node_group" "photonic_gpu_nodes" {
  cluster_name    = aws_eks_cluster.photonic_cluster.name
  node_group_name = "photonic-gpu-${var.environment}"
  node_role_arn   = aws_iam_role.photonic_node_role.arn
  subnet_ids      = aws_subnet.photonic_private_subnets[*].id

  capacity_type  = "ON_DEMAND"
  instance_types = var.gpu_node_instance_types

  scaling_config {
    desired_size = var.gpu_node_desired_size
    max_size     = var.gpu_node_max_size
    min_size     = var.gpu_node_min_size
  }

  update_config {
    max_unavailable = 1
  }

  labels = {
    role = "photonic"
    accelerator = "photonic"
  }

  taint {
    key    = "photonic-node"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  depends_on = [
    aws_iam_role_policy_attachment.photonic_node_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.photonic_node_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.photonic_node_AmazonEC2ContainerRegistryReadOnly,
  ]

  tags = {
    Name = "photonic-gpu-nodes-${var.environment}"
  }
}

# ECR Repository for container images
resource "aws_ecr_repository" "photonic_repo" {
  name                 = "photonic-flash-attention"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Name = "photonic-flash-attention-repo"
  }
}

# ECR Lifecycle Policy
resource "aws_ecr_lifecycle_policy" "photonic_repo_policy" {
  repository = aws_ecr_repository.photonic_repo.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 production images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Delete untagged images older than 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# S3 Bucket for application data and backups
resource "aws_s3_bucket" "photonic_data" {
  bucket        = "photonic-flash-attention-${var.environment}-${random_id.bucket_suffix.hex}"
  force_destroy = var.environment != "production"

  tags = {
    Name = "photonic-data-${var.environment}"
  }
}

resource "aws_s3_bucket_versioning" "photonic_data_versioning" {
  bucket = aws_s3_bucket.photonic_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "photonic_data_encryption" {
  bucket = aws_s3_bucket.photonic_data.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "photonic_data_pab" {
  bucket = aws_s3_bucket.photonic_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Random ID for unique S3 bucket naming
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Outputs
output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = aws_eks_cluster.photonic_cluster.endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.photonic_cluster.vpc_config[0].cluster_security_group_id
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.photonic_cluster.arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.photonic_cluster.certificate_authority[0].data
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.photonic_repo.repository_url
}

output "s3_bucket_name" {
  description = "S3 bucket name for application data"
  value       = aws_s3_bucket.photonic_data.bucket
}