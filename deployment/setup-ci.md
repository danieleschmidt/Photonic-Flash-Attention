# GitHub Actions CI/CD Setup

Due to GitHub App permissions, the CI workflow file needs to be manually added to your repository. Follow these steps:

## 1. Add GitHub Actions Workflow

Copy the workflow file to your repository:

```bash
# Create .github/workflows directory
mkdir -p .github/workflows

# Copy the CI workflow
cp deployment/github-actions-ci.yml .github/workflows/ci.yml

# Commit and push
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow"
git push
```

## 2. Configure Secrets

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

### Required Secrets
- `TEST_PYPI_API_TOKEN`: Token for publishing to Test PyPI
- `CODECOV_TOKEN`: Token for code coverage reporting (optional)

### How to Get Tokens

**Test PyPI Token:**
1. Create an account at https://test.pypi.org/
2. Go to Account Settings → API tokens
3. Create a new token with appropriate scope
4. Add it as a repository secret

**Codecov Token:**
1. Sign up at https://codecov.io/ with your GitHub account
2. Add your repository
3. Copy the upload token
4. Add it as a repository secret

## 3. Enable Actions

Make sure GitHub Actions are enabled in your repository:
1. Go to Settings → Actions → General
2. Select "Allow all actions and reusable workflows"
3. Save

## 4. Verify Setup

Once the workflow is added:
1. Push a commit to trigger the workflow
2. Check the Actions tab to see the pipeline running
3. Verify all jobs pass successfully

## 5. Branch Protection (Optional)

To enforce CI checks before merging:
1. Go to Settings → Branches
2. Add a rule for your main branch
3. Require status checks to pass before merging
4. Select the CI workflow jobs

## Alternative CI Providers

If you prefer other CI providers, we also provide configurations for:

### GitLab CI
See `deployment/gitlab-ci.yml`

### Jenkins
See `deployment/Jenkinsfile`

### CircleCI
See `deployment/circle-ci.yml`

## Local Testing

Before pushing, you can test the workflow locally using `act`:

```bash
# Install act
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run the workflow locally
act push
```

## Troubleshooting

**Workflow not running?**
- Check that Actions are enabled in repository settings
- Verify the workflow file is in `.github/workflows/`
- Check the workflow syntax

**Tests failing?**
- Run tests locally first: `pytest tests/`
- Check Python version compatibility
- Verify all dependencies are installed

**Deployment failing?**
- Check that secrets are correctly configured
- Verify token permissions and scope
- Test PyPI upload manually first