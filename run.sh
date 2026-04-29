#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Rust install + build + run (Ubuntu / WSL)
# ------------------------------------------------------------
# Ce script :
#  - détecte automatiquement le dossier projet à partir de ce run.sh
#  - vérifie que le nom du dossier est valide pour Cargo
#  - lit [package].name depuis Cargo.toml (fallback : nom du dossier)
#  - installe gcc/curl/rust si nécessaire
#  - compile en release puis lance le binaire
# ============================================================

# ------------------------------------------------------------
# Détection automatique du dossier projet = dossier du script
# ------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$SCRIPT_DIR"
PROJECT_NAME="$(basename -- "$PROJECT_DIR")"
CARGO_TOML="$PROJECT_DIR/Cargo.toml"

# ------------------------------------------------------------
# Vérification du nom du dossier
# Autorisé : a-z et 0-9 seulement
# (au moins 1 caractère)
# ------------------------------------------------------------
if [[ ! "$PROJECT_NAME" =~ ^[a-z0-9]+$ ]]; then
    echo "[error] Nom de dossier invalide : $PROJECT_NAME"
    echo "[error] Autorisé uniquement : a-z et 0-9"
    echo "[error] Exemples valides : projet1, p8, sieve9, projet10"
    echo "[error] Exemples invalides : Projet1, projet_1, projet-1"
    exit 1
fi

# ------------------------------------------------------------
# Vérifier que le projet existe
# ------------------------------------------------------------
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "[error] Répertoire projet introuvable : $PROJECT_DIR"
    exit 1
fi

if [[ ! -f "$CARGO_TOML" ]]; then
    echo "[error] Cargo.toml introuvable dans $PROJECT_DIR"
    exit 1
fi

# ------------------------------------------------------------
# Lire le nom du binaire depuis Cargo.toml
# Priorité : [[bin]].name → [package].name → nom du dossier
# ------------------------------------------------------------
read_section_name() {
    local section="$1"
    awk -v section="$section" '
        BEGIN { in_section = 0 }
        $0 == "[" section "]" || $0 == "[[" section "]]" { in_section = 1; next }
        /^\[/ { in_section = 0 }
        in_section && /^[[:space:]]*name[[:space:]]*=/ {
            line = $0
            sub(/^[[:space:]]*name[[:space:]]*=[[:space:]]*"/, "", line)
            sub(/".*$/, "", line)
            print line
            exit
        }
    ' "$CARGO_TOML"
}

read_cargo_name() {
    local bin_name
    bin_name="$(read_section_name 'bin')"
    if [[ -n "$bin_name" ]]; then
        PACKAGE_NAME="$bin_name"
        echo "[info] Binary : $PACKAGE_NAME (lu depuis [[bin]] dans Cargo.toml)"
        return
    fi

    local pkg_name
    pkg_name="$(read_section_name 'package')"
    if [[ -n "$pkg_name" ]]; then
        PACKAGE_NAME="$pkg_name"
        echo "[info] Binary : $PACKAGE_NAME (lu depuis [package] dans Cargo.toml)"
        return
    fi

    echo "[warn] Aucun name trouvé dans Cargo.toml, fallback sur le nom du dossier : $PROJECT_NAME"
    PACKAGE_NAME="$PROJECT_NAME"
}

read_cargo_name

# ------------------------------------------------------------
# Assurer gcc (libc)
# ------------------------------------------------------------
if ! command -v gcc >/dev/null 2>&1; then
    echo "[info] gcc non détecté, installation…"
    sudo apt update -y
    sudo apt install -y gcc
fi

# ------------------------------------------------------------
# Assurer curl (pour rustup)
# ------------------------------------------------------------
if ! command -v curl >/dev/null 2>&1; then
    echo "[info] curl non détecté, installation…"
    sudo apt update -y
    sudo apt install -y curl
fi

# ------------------------------------------------------------
# Assurer Rust (cargo + rustc)
# ------------------------------------------------------------
ensure_rust() {
    # Charger ~/.cargo/bin dans le PATH même en bash non-interactif
    # (sinon command -v cargo échoue et on retente une install rustup).
    export PATH="$HOME/.cargo/bin:$PATH"

    if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
        return
    fi

    echo "[info] Rust non détecté, installation via rustup…"

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

    export PATH="$HOME/.cargo/bin:$PATH"

    if ! grep -q 'export PATH="$HOME/.cargo/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
        echo "" >> "$HOME/.bashrc"
        echo "# Rust (rustup)" >> "$HOME/.bashrc"
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.bashrc"
        echo "[info] PATH Rust ajouté à ~/.bashrc"
    fi

    if ! command -v cargo >/dev/null 2>&1; then
        echo "[error] L'installation de Rust a échoué"
        exit 1
    fi

    echo "[info] Rust installé avec succès"
}

ensure_rust

# ------------------------------------------------------------
# Se placer dans le projet
# ------------------------------------------------------------
cd "$PROJECT_DIR"

# ------------------------------------------------------------
# Compilation release avec CPU natif
# ------------------------------------------------------------
export RUSTFLAGS="-C target-cpu=native"
cargo build --release

# ------------------------------------------------------------
# Détection du nom du binaire (lu depuis Cargo.toml)
# ------------------------------------------------------------
BIN="$PACKAGE_NAME"
EXE="./target/release/$BIN"

if [[ ! -x "$EXE" ]]; then
    echo "[error] exécutable introuvable: $EXE"
    exit 1
fi

# ------------------------------------------------------------
# Lancer le programme (arguments passés au script)
# ------------------------------------------------------------
"$EXE" "$@"
