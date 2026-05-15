#!/usr/bin/env bash
set -euo pipefail

DB_NAME="${DEEPBINDER_POSTGRES_DB:-deepbinder}"
DB_USER="${DEEPBINDER_POSTGRES_USER:-deepbinder}"
DB_PASSWORD="${DEEPBINDER_POSTGRES_PASSWORD:-deepbinder_change_me}"

if ! command -v apt-get >/dev/null 2>&1; then
    echo "当前脚本支持 Ubuntu/Debian apt 环境。请手动安装 PostgreSQL 后设置 DEEPBINDER_DATABASE_URL。"
    exit 1
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "请用 root 执行，或使用 sudo: sudo -E bash scripts/install_postgresql.sh"
    exit 1
fi

apt-get update
apt-get install -y postgresql postgresql-contrib
systemctl enable postgresql
systemctl start postgresql

sudo -u postgres psql <<SQL
DO \$\$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${DB_USER}') THEN
      CREATE ROLE ${DB_USER} LOGIN PASSWORD '${DB_PASSWORD}';
   ELSE
      ALTER ROLE ${DB_USER} WITH LOGIN PASSWORD '${DB_PASSWORD}';
   END IF;
END
\$\$;

SELECT 'CREATE DATABASE ${DB_NAME} OWNER ${DB_USER}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${DB_NAME}')\gexec
GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};
SQL

echo "PostgreSQL 已准备完成。后端启动前设置："
echo "export DEEPBINDER_DATABASE_URL='postgresql+psycopg://${DB_USER}:${DB_PASSWORD}@127.0.0.1:5432/${DB_NAME}'"
