Run SQL scripts against a LOCAL COPY of the database (e.g., polymarket\_work.db), not the original.

Scripts use BEGIN TRANSACTION; review sanity checks; then COMMIT; otherwise ROLLBACK.

Never commit \*.db files to GitHub (see .gitignore).

