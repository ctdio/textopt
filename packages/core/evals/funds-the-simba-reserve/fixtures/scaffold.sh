set -euo pipefail
REPO=/home/ctdio/projects/open-source/textopt
CASE="$REPO/packages/core/evals/funds-the-simba-reserve"
if [ ! -d "$REPO/packages/core/dist" ]; then
  echo "textopt is not built; run: npm --prefix $REPO/packages/core run build" >&2
  exit 1
fi
mkdir -p node_modules/textopt out
cp -R "$REPO/packages/core/dist" "$REPO/packages/core/docs" node_modules/textopt/
cp "$REPO/packages/core/package.json" "$REPO/packages/core/README.md" node_modules/textopt/
cp "$CASE/fixtures/data.mjs" "$CASE/fixtures/config.mjs" "$CASE/fixtures/run.mjs" "$CASE/fixtures/verify.mjs" .
printf '{"name":"simba-run","private":true,"type":"module"}\n' > package.json
