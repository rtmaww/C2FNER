C2FNER
=============
Codes for ACL 2023 findings paper "Coarse-to-fine Few-shot Learning for Named Entity Recognition"

Dependencies:
------------
python 3.8.5

cuda 11.0

To install the required packages by following commands:
>$ pip install -r requirements.txt

To download the pretrained bert-base-cased model:
>$ cd bert-base-cased/
> 
>$ sh download_bert.sh

Coarse training
-----------
First run cluster.py to get the clustering result:
>$ python cluster.py

then run get_proto_from_clusters.py to get the prototype of the clustering result:
>$ python get_proto_from_clusters.py

finally run scripts/run_cluster.sh
>$ bash scripts/run_cluster.sh

Fine training
---------
First run get_base_statistics.py:
>$ python get_base_statistics.py

then run get_calibration.py:
>$ python get_calibration.py

finally run scripts/run_few.sh
>$ bash scripts/run_few.sh


