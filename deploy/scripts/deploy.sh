#!/bin/bash
set -euo pipefail

# Photonic Flash Attention - Production Deployment Script
# This script handles the complete deployment pipeline

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_DIR="$PROJECT_ROOT/deploy"

# Default values
ENVIRONMENT="${ENVIRONMENT:-dev}"
AWS_REGION="${AWS_REGION:-us-west-2}"
CLUSTER_NAME="photonic-cluster-${ENVIRONMENT}"
NAMESPACE="photonic-system"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Error handling
error_exit() {
    log_error "$1"
    exit 1
}

# Help function
show_help() {
    cat << EOF
Photonic Flash Attention Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
  infrastructure    Deploy AWS infrastructure with Terraform
  build            Build and push Docker images
  deploy           Deploy application to Kubernetes
  all              Run infrastructure, build, and deploy
  destroy          Destroy infrastructure (use with caution)
  status           Check deployment status

Options:
  -e, --environment ENV    Environment (dev, staging, production) [default: dev]
  -r, --region REGION      AWS region [default: us-west-2]
  -t, --tag TAG           Docker image tag [default: latest]
  -h, --help              Show this help message

Environment Variables:
  AWS_PROFILE             AWS profile to use
  KUBECONFIG             Path to kubeconfig file
  DOCKER_REGISTRY        Docker registry URL

Examples:
  $0 -e production infrastructure
  $0 -e staging build
  $0 -e dev deploy
  $0 all
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -r|--region)
                AWS_REGION="$2"
                shift 2
                ;;
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                if [[ -z "${COMMAND:-}" ]]; then
                    COMMAND="$1"
                else
                    error_exit "Unknown argument: $1"
                fi
                shift
                ;;
        esac
    done

    if [[ -z "${COMMAND:-}" ]]; then
        error_exit "No command specified. Use -h for help."
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    for tool in aws kubectl terraform docker; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error_exit "Missing required tools: ${missing_tools[*]}"
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error_exit "AWS credentials not configured or invalid"
    fi
    
    log_success "Prerequisites check passed"
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    log_info "Deploying infrastructure for environment: $ENVIRONMENT"
    
    cd "$DEPLOY_DIR/terraform"
    
    # Initialize Terraform
    terraform init -backend-config="bucket=terragon-terraform-state-${ENVIRONMENT}" \
                   -backend-config="key=photonic-flash-attention/terraform.tfstate" \
                   -backend-config="region=${AWS_REGION}"
    
    # Plan and apply
    terraform plan \
        -var="environment=${ENVIRONMENT}" \
        -var="aws_region=${AWS_REGION}" \
        -out="terraform.plan"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_warning "Production deployment detected. Please review the plan above."
        read -p "Continue with production deployment? (yes/no): " confirm
        if [[ "$confirm" != "yes" ]]; then
            log_info "Deployment cancelled by user"
            exit 0
        fi
    fi
    
    terraform apply "terraform.plan"
    
    # Export cluster info for kubectl configuration
    CLUSTER_ARN=$(terraform output -raw cluster_arn)
    ECR_REPOSITORY_URL=$(terraform output -raw ecr_repository_url)
    
    # Update kubeconfig
    aws eks update-kubeconfig \
        --region "$AWS_REGION" \
        --name "$CLUSTER_NAME" \
        --alias "$CLUSTER_NAME"
    
    log_success "Infrastructure deployment completed"
}

# Build and push Docker images
build_and_push() {
    log_info "Building and pushing Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Get ECR repository URL from Terraform output
    if [[ -f "$DEPLOY_DIR/terraform/terraform.tfstate" ]]; then
        cd "$DEPLOY_DIR/terraform"
        ECR_REPOSITORY_URL=$(terraform output -raw ecr_repository_url 2>/dev/null || echo "")
        cd "$PROJECT_ROOT"
    fi
    
    if [[ -z "${ECR_REPOSITORY_URL:-}" ]]; then
        if [[ -z "${DOCKER_REGISTRY:-}" ]]; then
            error_exit "No ECR repository URL found and DOCKER_REGISTRY not set"
        fi
        ECR_REPOSITORY_URL="$DOCKER_REGISTRY/photonic-flash-attention"
    fi
    
    # Login to ECR
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$(echo "$ECR_REPOSITORY_URL" | cut -d'/' -f1)"
    
    # Build Docker image
    docker build \
        -f "$DEPLOY_DIR/docker/Dockerfile" \
        -t "photonic-flash-attention:$IMAGE_TAG" \
        -t "$ECR_REPOSITORY_URL:$IMAGE_TAG" \
        --target production \
        .
    
    # Push to registry
    docker push "$ECR_REPOSITORY_URL:$IMAGE_TAG"
    
    log_success "Docker image built and pushed: $ECR_REPOSITORY_URL:$IMAGE_TAG"
}

# Deploy application to Kubernetes
deploy_application() {
    log_info "Deploying application to Kubernetes..."
    
    # Ensure kubectl is configured for the right cluster
    kubectl config use-context "$CLUSTER_NAME"
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Update deployment with current image
    if [[ -n "${ECR_REPOSITORY_URL:-}" ]]; then
        sed "s|photonic-flash-attention:latest|${ECR_REPOSITORY_URL}:${IMAGE_TAG}|g" \
            "$DEPLOY_DIR/kubernetes/deployment.yaml" > "/tmp/deployment-${ENVIRONMENT}.yaml"
    else
        cp "$DEPLOY_DIR/kubernetes/deployment.yaml" "/tmp/deployment-${ENVIRONMENT}.yaml"
    fi
    
    # Apply Kubernetes manifests
    kubectl apply -f "/tmp/deployment-${ENVIRONMENT}.yaml"
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/photonic-flash-attention \
        -n "$NAMESPACE" \
        --timeout=300s
    
    log_success "Application deployed successfully"
}

# Check deployment status
check_status() {
    log_info "Checking deployment status..."
    
    # Infrastructure status
    if [[ -f "$DEPLOY_DIR/terraform/terraform.tfstate" ]]; then
        log_info "Infrastructure Status:"
        cd "$DEPLOY_DIR/terraform"
        terraform show -json | jq -r '.values.root_module.resources[] | select(.type == "aws_eks_cluster") | .values.name'
        cd "$PROJECT_ROOT"
    else
        log_warning "No Terraform state found"
    fi
    
    # Kubernetes status
    if kubectl config current-context &> /dev/null; then
        log_info "Kubernetes Status:"
        kubectl get deployment,service,pod -n "$NAMESPACE" 2>/dev/null || log_warning "Namespace $NAMESPACE not found"
    else
        log_warning "kubectl not configured"
    fi
}

# Destroy infrastructure
destroy_infrastructure() {
    log_warning "DESTRUCTIVE OPERATION: This will destroy all infrastructure"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_error "Production destruction requires manual confirmation"
        read -p "Type 'DESTROY PRODUCTION' to confirm: " confirm
        if [[ "$confirm" != "DESTROY PRODUCTION" ]]; then
            log_info "Destruction cancelled"
            exit 0
        fi
    else
        read -p "Destroy $ENVIRONMENT environment? (yes/no): " confirm
        if [[ "$confirm" != "yes" ]]; then
            log_info "Destruction cancelled"
            exit 0
        fi
    fi
    
    cd "$DEPLOY_DIR/terraform"
    
    terraform destroy \
        -var="environment=${ENVIRONMENT}" \
        -var="aws_region=${AWS_REGION}" \
        -auto-approve
    
    log_success "Infrastructure destroyed"
}

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        dev|staging|production)
            ;;
        *)
            error_exit "Invalid environment: $ENVIRONMENT. Must be dev, staging, or production."
            ;;
    esac
}

# Main execution
main() {
    log_info "Photonic Flash Attention Deployment Script"
    log_info "Environment: $ENVIRONMENT"
    log_info "Region: $AWS_REGION"
    log_info "Image Tag: $IMAGE_TAG"
    
    validate_environment
    check_prerequisites
    
    case $COMMAND in
        infrastructure)
            deploy_infrastructure
            ;;
        build)
            build_and_push
            ;;
        deploy)
            deploy_application
            ;;
        all)
            deploy_infrastructure
            build_and_push
            deploy_application
            check_status
            ;;
        status)
            check_status
            ;;
        destroy)
            destroy_infrastructure
            ;;
        *)
            error_exit "Unknown command: $COMMAND"
            ;;
    esac
    
    log_success "Operation completed successfully!"
}

# Parse arguments and run
parse_args "$@"
main