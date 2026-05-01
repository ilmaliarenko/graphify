terraform {
  required_version = ">= 1.5.0"
}

provider "aws" {
  region = var.region
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "environment" {
  type = string
}

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  bucket_prefix = "${var.environment}-data"
}

module "logging_bucket" {
  source      = "./modules/s3"
  bucket_name = "${local.bucket_prefix}-logs"
  environment = var.environment
}

resource "aws_iam_role" "lambda_exec" {
  name = "${var.environment}-lambda-exec"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_lambda_function" "processor" {
  function_name = "${var.environment}-processor"
  role          = aws_iam_role.lambda_exec.arn
  handler       = "index.handler"
  runtime       = "python3.12"

  environment {
    variables = {
      LOG_BUCKET = module.logging_bucket.bucket_name
      ACCOUNT_ID = local.account_id
    }
  }
}

output "lambda_arn" {
  value = aws_lambda_function.processor.arn
}
