# Guía Rápida de Git

## ✅ Estado Actual
- ✅ Repositorio Git inicializado
- ✅ Primer commit realizado (versión inicial)
- ✅ Archivos importantes guardados

## 📝 Comandos Básicos de Git

### Ver el estado del repositorio
```bash
git status
```

### Ver el historial de commits
```bash
git log --oneline
```

### Agregar archivos modificados
```bash
git add detector_objetos_oakd.py
# O agregar todos los archivos modificados:
git add .
```

### Hacer un commit (guardar cambios)
```bash
git commit -m "Descripción de los cambios realizados"
```

### Ver qué archivos cambiaron
```bash
git diff
```

## 🔄 Flujo de Trabajo Típico

1. **Hacer cambios** en los archivos
2. **Ver qué cambió**: `git status`
3. **Agregar cambios**: `git add .` o `git add archivo_especifico.py`
4. **Hacer commit**: `git commit -m "Descripción de cambios"`
5. **Verificar**: `git log --oneline`

## 📋 Ejemplos de Mensajes de Commit

```bash
# Agregar nueva funcionalidad
git commit -m "Agregar detección de obstáculos próximos"

# Corregir un error
git commit -m "Corregir problema con síntesis de voz"

# Mejorar código existente
git commit -m "Mejorar rendimiento de detección"

# Cambio menor
git commit -m "Actualizar README con nuevas instrucciones"
```

## 🔍 Comandos Útiles

### Ver cambios antes de hacer commit
```bash
git diff
```

### Deshacer cambios en un archivo (antes de hacer commit)
```bash
git checkout -- nombre_archivo.py
```

### Ver el historial completo
```bash
git log
```

### Ver un commit específico
```bash
git show 7af5872
```

## ⚠️ Archivos que NO se guardan (están en .gitignore)
- `venv/` - Entorno virtual
- `*.pt` - Modelos YOLO (son muy grandes)
- `__pycache__/` - Archivos temporales de Python
- `*.log` - Archivos de log

## 💡 Consejos
- Haz commits frecuentes (cada vez que agregues una funcionalidad o corrijas un error)
- Usa mensajes de commit descriptivos
- Revisa `git status` antes de hacer commit para ver qué se va a guardar

