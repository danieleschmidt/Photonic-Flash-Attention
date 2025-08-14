# Terraform variables for Photonic Flash Attention infrastructure

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "dev"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

variable "nat_gateway_count" {
  description = "Number of NAT Gateways to create"
  type        = number
  default     = 1
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.27"
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "CIDR blocks that can access the EKS cluster endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "cloudwatch_log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 14
}

# General node group variables
variable "general_node_instance_types" {
  description = "Instance types for general node group"
  type        = list(string)
  default     = ["m5.large", "m5.xlarge"]
}

variable "general_node_desired_size" {
  description = "Desired number of nodes in general node group"
  type        = number
  default     = 2
}

variable "general_node_min_size" {
  description = "Minimum number of nodes in general node group"
  type        = number
  default     = 1
}

variable "general_node_max_size" {
  description = "Maximum number of nodes in general node group"
  type        = number
  default     = 10
}

# GPU node group variables
variable "gpu_node_instance_types" {
  description = "Instance types for GPU node group"
  type        = list(string)
  default     = ["p3.2xlarge", "p3.8xlarge"]
}

variable "gpu_node_desired_size" {
  description = "Desired number of nodes in GPU node group"
  type        = number
  default     = 0
}

variable "gpu_node_min_size" {
  description = "Minimum number of nodes in GPU node group"
  type        = number
  default     = 0
}

variable "gpu_node_max_size" {
  description = "Maximum number of nodes in GPU node group"
  type        = number
  default     = 5
}

# Tagging
variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Project     = "photonic-flash-attention"
    ManagedBy   = "terraform"
    Owner       = "terragon-labs"
  }
}