FROM gridworld
RUN mkdir /opt/rlshield
WORKDIR /opt/rlshield


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python setup.py install
