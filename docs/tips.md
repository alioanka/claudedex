For Python Changes ONLY:
bash# Quick restart - NO rebuild needed
docker-compose restart dashboard

# Or if you want to see logs immediately
docker-compose restart dashboard && docker-compose logs -f dashboard
When You NEED to Rebuild:
You only need docker-compose build when:

❌ Changes to Dockerfile
❌ Changes to requirements.txt (new packages)
❌ Changes to docker-compose.yml

Optimized Workflow by File Type:
For Python files (.py):
git pull origin main
docker-compose restart dashboard
docker-compose logs -f dashboard
For Frontend files (.html, .js, .css):
git pull origin main
# NO restart needed! Just refresh browser (Ctrl+F5)
For requirements.txt or Dockerfile:
git pull origin main
docker-compose down
docker-compose build
docker-compose up -d
docker-compose logs -f dashboard
Even Better - Hot Reload Setup:
Add this to your docker-compose.yml to enable auto-reload:
yamlservices:
  dashboard:
    # ... existing config ...
    volumes:
      - ./monitoring:/app/monitoring:ro
      - ./dashboard:/app/dashboard:ro
      - ./config:/app/config:ro
      - ./data:/app/data:ro
      - ./core:/app/core:ro
    environment:
      - PYTHONUNBUFFERED=1
Then you won't need to restart at all - changes will auto-reload!
Quick Reference:
File ChangedCommand.py filesdocker-compose restart dashboard.html/.js/.cssJust refresh browserrequirements.txtdocker-compose buildDockerfiledocker-compose build
Your Current Workflow Simplified:
bash# 1. On local: push changes
git push origin main

# 2. On VPS: pull and quick restart
git pull origin main
docker-compose restart dashboard
docker-compose logs -f dashboard

# Press Ctrl+C to stop watching logs
Save 90% of time by skipping the build step! 🚀

