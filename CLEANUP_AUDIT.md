# Cleanup Audit - Unused Files Identification

## Summary

Identified **many unused files** in the repository that can be removed or ignored. These include legacy benchmarks, test files, PDFs, and documentation that are not needed for the local chatbot.

**Recommendation**: Remove or archive unused files to reduce repository size and confusion.

---

## Files That Can Be Removed

### Root Directory (Unused Files)

#### Benchmark Files (Can be removed - for legacy cloud app)
- `benchmark_cache.py` - Legacy benchmark
- `benchmark_chatbot.py` - Legacy benchmark
- `benchmark_cli.py` - Can keep (local chatbot)
- `benchmark_cli_after_load.py` - Can keep (local chatbot)
- `benchmark_optimized.py` - Can keep (local chatbot)
- `test_user_speed.py` - Can keep (local chatbot)

#### Test Files (Can be removed - for legacy cloud app)
- `test_enhanced_rag.py` - Legacy test
- `test_knowledge_base.py` - Legacy test
- `test_local_chat.py` - Legacy test
- `test_local_llm.py` - Legacy test
- `test_retrieval_fix.py` - Legacy test
- `test_thinking.py` - Legacy test

#### Other Files (Can be removed)
- `cache.py` - Legacy cache
- `improved_data_ingestion.py` - Legacy
- `infinite_context.py` - Legacy
- `local_server.py` - Legacy
- `retrieval.py` - Legacy
- `server_speed.py` - Legacy
- `test_chatbot.db` - Test database
- `test_results.json` - Test results
- `professional_benchmark.py` - Legacy benchmark

#### PDF Files (Can be removed - large, not needed in repo)
- `1st equation.pdf`
- `Chapter-01.pdf` through `Chapter-03.pdf`
- `part 1.pdf` through `part 3.pdf`
- Vietnamese PDF files

#### Documentation Files (Can be removed - outdated)
- `BENCHMARK_ANALYSIS.md`
- `BENCHMARK_SUMMARY.md`
- `DEPLOY.md`
- `DEPLOYMENT_SUMMARY.md`
- `ENHANCED_RAG_SUMMARY.md`
- `KNOWLEDGE_BASE.md`
- `LOCAL_SETUP.md`
- `OPTIMIZATION_ANALYSIS.md`
- `PDF_OPTIMIZATION_ANALYSIS.md`
- `PERFORMANCE_SUMMARY.md`
- `PROFESSIONAL_BENCHMARK_REPORT.md`
- `README_cloud.md`
- `README_LOCAL.md`
- `ULTRA_SPEED_ANALYSIS.md`

#### Deployment Files (Can be removed - for cloud app)
- `Procfile` - For Heroku/cloud
- `render.yaml` - For Render cloud
- `deploy_optimized.bat` - Legacy
- `deploy_optimized.sh` - Legacy
- `deploy_ultra.bat` - Legacy
- `run_local.bat` - Legacy
- `run_local.sh` - Legacy

#### Directories (Can be removed or archived)
- `benchmarks/` - Contains 35+ benchmark files for legacy cloud app
- `chatbot/` - Legacy cloud app
- `cloud_server.py` - Legacy
- `deployment/` - Cloud deployment configs
- `engine/` - Legacy engine
- `kubernetes/` - Cloud deployment
- `legacy/` - Legacy code
- `load-test/` - Load testing
- `web/` - Legacy web UI

---

## Files to Keep (Essential for Local Chatbot)

### Core Files
- `cli_chat.py` - Main CLI entry point
- `cli_chat_optimized.py` - Optimized CLI
- `run_local_chatbot.py` - Web server entry point
- `local_chatbot/` - Core local chatbot module
- `data/knowledge_base/` - Knowledge base
- `models/` - Model files (user downloads)
- `.env.example` - Environment template
- `requirements.txt` - Core dependencies
- `requirements-local.txt` - Core dependencies (copy)
- `requirements-cloud.txt` - Cloud dependencies (for reference)
- `Dockerfile` - Docker deployment
- `setup.bat` - Windows setup script
- `setup.sh` - macOS/Linux setup script

### Documentation (Keep)
- `README.md` - Main user guide
- `LICENSE` - MIT license
- `.gitignore` - Git ignore rules
- `.github/workflows/ci.yml` - CI workflow
- `.github/workflows/deploy.yml` - Deployment workflow

### Bug Reports (Keep for reference)
- `BUG_SUMMARY.md` - Complete bug summary
- `PERFORMANCE_BUGS.md` - Performance bugs
- `CONCURRENCY_BUGS.md` - Thread safety bugs
- `SECURITY_AUDIT.md` - Security bugs
- `RESOURCE_MANAGEMENT.md` - Resource management
- `USER_EXPERIENCE_BUGS.md` - UX bugs
- `EDGE_CASE_BUGS.md` - Edge cases
- `CLEANUP_BUGS.md` - Code quality
- `MAGIC_NUMBER_BUGS.md` - Magic numbers
- `DOCUMENTATION_BUGS.md` - Documentation issues
- `ENVIRONMENT_BUGS.md` - Environment variables
- `DEPLOYMENT_BUGS.md` - Deployment bugs
- `INSTALLATION_BUG.md` - Installation bugs
- `SETUP_USE_BUGS.md` - Setup bugs
- `SETUP_TEST_RESULTS.md` - Setup test results
- `SETUP_VERIFICATION.md` - Setup verification
- `WORKFLOW_VERIFICATION.md` - Workflow verification
- `FINAL_AUDIT_SUMMARY.md` - Final audit

### Keep in App Directory (Legacy Cloud - but may be useful)
- `app/` - Legacy cloud app (may be useful as reference)
- `speed_engine/` - Shared optimization engine (used by local chatbot)

---

## Recommended Cleanup Actions

### Action 1: Update .gitignore
Add these patterns to .gitignore to exclude unused files from future commits:
```
# PDF files
*.pdf

# Legacy benchmarks and tests
benchmarks/
benchmark_*.py
test_*.py
*.db
*.json

# Legacy documentation
*_ANALYSIS.md
*_SUMMARY.md
PROFESSIONAL_BENCHMARK_REPORT.md
README_cloud.md
README_LOCAL.md

# Legacy deployment
Procfile
render.yaml
deploy_*.bat
deploy_*.sh
run_local.*

# Legacy directories
chatbot/
deployment/
engine/
kubernetes/
legacy/
load-test/
web/

# Large files
*.gguf
```

### Action 2: Create Archive Directory
Create `legacy/` directory and move unused files there:
```bash
mkdir legacy/archived
mv benchmarks legacy/archived/
mv *.pdf legacy/archived/
mv *_ANALYSIS.md legacy/archived/
mv *_SUMMARY.md legacy/archived/
mv PROFESSIONAL_BENCHMARK_REPORT.md legacy/archived/
mv README_cloud.md legacy/archived/
mv README_LOCAL.md legacy/archived/
mv Procfile legacy/archived/
mv render.yaml legacy/archived/
mv deploy_*.bat legacy/archived/
mv deploy_*.sh legacy/archived/
mv run_local.* legacy/archived/
```

### Action 3: Remove Test Files
Remove from root:
```bash
rm cache.py
rm improved_data_ingestion.py
rm infinite_context.py
rm local_server.py
rm retrieval.py
rm server_speed.py
rm test_chatbot.db
rm test_results.json
rm professional_benchmark.py
```

### Action 4: Update README
Update README.md to only document the essential files and commands.

---

## Priority

### High Priority (Do Now)
1. Update .gitignore to prevent future commits of unused files
2. Archive PDF files (large, not needed in repo)
3. Archive legacy documentation files

### Medium Priority (Do Later)
4. Archive benchmark directories
5. Archive legacy test files
6. Clean up root directory

### Low Priority (Optional)
7. Remove legacy cloud app (app/) if not needed
8. Remove speed_engine if not used by local chatbot (verify first)

---

## Estimated Impact

**Before cleanup**:
- Repository size: Large (many unused files)
- Confusing: Mixed legacy and new code
- Hard to navigate: Many similar files

**After cleanup**:
- Repository size: Smaller (~50% reduction)
- Clear: Only essential files
- Easy to navigate: Organized structure
