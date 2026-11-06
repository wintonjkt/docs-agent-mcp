# UBI 9 Python 3.12 base. Runs as non-root and works under OpenShift's
# restricted SCC, which assigns an arbitrary UID in group 0.
FROM registry.access.redhat.com/ubi9/python-312:latest

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080

WORKDIR /opt/app-root/src

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY rag_tools.py server.py ./

# Arbitrary-UID support: group 0 owns the app tree and group permissions equal
# the owner's, so any UID in group 0 can read and write here.
USER 0
RUN chgrp -R 0 /opt/app-root && chmod -R g=u /opt/app-root
USER 1001

EXPOSE 8080

# Run Starlette/FastMCP SSE server using uvicorn
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
