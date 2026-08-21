set -euo pipefail
REPO=/home/ctdio/projects/open-source/textopt
CASE="$REPO/packages/core/evals/refuses-a-saturated-run"
if [ ! -d "$REPO/packages/core/dist" ]; then
  echo "textopt is not built; run: npm --prefix $REPO/packages/core run build" >&2
  exit 1
fi
mkdir -p node_modules/textopt out
cp -R "$REPO/packages/core/dist" "$REPO/packages/core/docs" node_modules/textopt/
cp "$REPO/packages/core/package.json" "$REPO/packages/core/README.md" node_modules/textopt/
cp "$CASE/fixtures/metric.mjs" "$CASE/fixtures/data.mjs" "$CASE/fixtures/run.mjs" "$CASE/fixtures/verify.mjs" .
printf 'Answer the customer question directly. Cite the relevant policy and keep the reply brief.\n' > prompt.txt
# pristine copies the oracle scores against, so a fix applied after the
# decision is recorded cannot move the number the decision is checked against
cp metric.mjs .pristine-metric.mjs
cp data.mjs .pristine-data.mjs
cp prompt.txt .pristine-prompt.txt
printf '{"name":"support-prompt","private":true,"type":"module"}\n' > package.json
