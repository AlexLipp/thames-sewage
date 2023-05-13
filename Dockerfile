FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.10-x86_64

RUN yum update -y && yum install -y wget tar bzip2 git && yum clean all

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

COPY micromamba-env.yml /tmp/micromamba-env.yml

RUN ./bin/micromamba env create --file /tmp/micromamba-env.yml --prefix /opt/conda-env

RUN mv /var/lang/bin/python3.10 /var/lang/bin/python3.10-clean && ln -sf /opt/conda-env/bin/python /var/lang/bin/python3.10

COPY src/sewage.py /var/task/sewage.py
COPY input_dir/mg_elev.obj  /var/task/input_dir/mg_elev.obj
COPY output_dir /var/task/output_dir

ENTRYPOINT [ "/opt/conda-env/bin/python", "-m", "awslambdaric" ]

CMD ["sewage.make_discharge_map"]