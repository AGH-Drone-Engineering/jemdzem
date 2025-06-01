
# Uruchomienie projektu `jemdzem` w kontenerze Docker

## 1. Zainstaluj i uruchom X-Server

Aby możliwe było uruchamianie aplikacji z interfejsem graficznym (GUI) w kontenerze, potrzebujesz lokalnego X-Servera.

**Proponowane narzędzie:** [VcXsrv (open-source)](https://sourceforge.net/projects/vcxsrv/)

### Jak znaleźć wartość `$DISPLAY`?

1. Uruchom VcXsrv (np. z domyślną konfiguracją: `Multiple windows`, `Start no client`, zaznacz `Disable access control`).
2. Po uruchomieniu kliknij **prawym przyciskiem myszy (PPM)** ikonę VcXsrv na pasku zadań.
3. Wybierz **"Show logs"**.
4. W logach znajdź fragment:

   ```
   DefineSelf - DESKTOP has <N> usable IPv4 interfaces...
    addresses X.X.X.X ...
   ```

5. Skopiuj **dowolny z podanych adresów IP** – posłuży jako wartość `DISPLAY`.

---

## 2. Zbuduj obraz Dockera

W katalogu z `Dockerfile` uruchom:

```bash
docker build -t jemdzem .
```

---

## 3. Uruchom kontener

Zamień `your-api-key` na Twój klucz Gemini API, a `X.X.X.X` na IP z logów VcXsrv:

```bash
docker run --name jemdzem_test -it --rm \
  -e GOOGLE_API_KEY=your-api-key \
  -e DISPLAY=X.X.X.X:0 \
  jemdzem
```

> **Uwaga:** Jeśli w VcXsrv zaznaczyłeś opcję `Disable access control`, nie musisz dodatkowo konfigurować `xhost`.

---

## 4. Połącz się z kontenerem przez VS Code

1. Upewnij się, że masz zainstalowane rozszerzenie **Remote - Containers**.
2. Otwórz **Visual Studio Code**.
3. Naciśnij `Ctrl+Shift+P` i wybierz:

   ```
   Remote-Containers: Attach to Running Container...
   ```

4. Wybierz kontener o nazwie `jemdzem_test`.
