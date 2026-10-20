# Pushing this folder to GitHub

The folder is already a git repository with the commits made so far (branch `main`).
Repository: https://github.com/pramothkanna/fairmallm-gan-thesis (private)

Open a terminal (Windows: right-click in the folder → *Open in Terminal*) and run:

```
git remote add origin https://github.com/pramothkanna/fairmallm-gan-thesis.git
git push -u origin main
```

If git says `remote origin already exists`, run `git remote set-url origin https://github.com/pramothkanna/fairmallm-gan-thesis.git`
and push again. Sign in to GitHub in the window that opens.

Later changes: `git add -A`, `git commit -m "short message"`, `git push`.

Never add: raw OULAD csv files, `Dataset/`, any ETB/BIBB file, `*.npy`, Word/PDF files (blocked by `.gitignore`).
