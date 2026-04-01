# MLOps Monitoring with FastAPI, Prometheus, and Grafana (работу выполнил Нестеров Дмитрий РИМ-240901)

**Проект для мониторинга FastAPI-приложения с использованием Prometheus и Grafana в Docker.**

---

## 📌 Описание проекта

Этот проект предоставляет инструменты для **мониторинга производительности** FastAPI-приложения:

- **FastAPI** — основное веб-приложение с поддержкой метрик для Prometheus.
- **Prometheus** — сбор, хранение и обработка метрик.
- **Grafana** — визуализация метрик, создание дашбордов и настройка алертов.

---

## 🛠 Структура проекта

```
/mlops
  ├── app/
  │   ├── main.py          # FastAPI приложение
  │   ├── Dockerfile       # Dockerfile для сборки FastAPI
  │   ├── requirements.txt # Зависимости Python
  │   └── prometheus.yml   # Конфигурация Prometheus
  └── docker-compose.yml   # Конфигурация Docker Compose
```

---

## 🚀 Быстрый старт

### Предварительные требования

- Установленный [Docker](https://docs.docker.com/get-docker/)
- Установленный [Docker Compose](https://docs.docker.com/compose/install/)

### Установка и запуск

1. Клонируйте репозиторий (если он есть):
  ```bash
   git clone <URL_вашего_репозитория>
   cd mlops
  ```
2. Запустите контейнеры:
  ```bash
   docker-compose up --build
  ```

---

## 🌐 Доступ к сервисам

- **FastAPI**: [http://localhost:8000](http://localhost:8000)
- **Prometheus**: [http://localhost:9090](http://localhost:9090)
- **Grafana**: [http://localhost:3000](http://localhost:3000) (логин/пароль: `admin/admin`)

---

## 📊 Мониторинг и метрики

### **Prometheus**

- Собирает метрики с FastAPI по эндпоинту `/metrics`.
- Конфигурация Prometheus находится в `app/prometheus.yml`.

### **Grafana**

- Подключается к Prometheus для визуализации метрик.
- Для подключения к Prometheus:
  1. Перейдите в **Configuration → Data Sources**.
  2. Добавьте источник данных **Prometheus** с URL: `http://prometheus:9090`.
---
