## Retriever 실행 방법

기본적으로 Search-R1의 bm25 retriever 환경을 사용합니다.

`/data3/seonhoyoo/multihopqa/Search-R1` 경로에 이미 Search-R1 repo가 클론되어 있고, retriever 환경이 설치되어 있습니다.

해당 경로에서 `sbatch retrieval_launch_bm25.sh` 를 실행하시면 됩니다.

### 주의사항

graphcheck 스크립트를 실행하는 노드와 동일한 노드에 retrieval_launch_bm25.sh 을 제출하셔야 합니다. 

sh 파일의 `#SBATCH --nodelist=**` 이 부분에서 노드 이름을 맞춰주세요.

### Retriever 환경 세팅 방법 (참고용)

아래 내용은 이미 제가 진행해 두어서 따로 실행하실 필요는 없습니다만, 참고로 적어둡니다.

1. Search-R1 repo clone 및 retriever environment 설치
    ```bash
    git clone https://github.com/PeterGriffinJin/Search-R1.git
    cd Search-R1

    # 출처: https://github.com/PeterGriffinJin/Search-R1
    conda create -n retriever python=3.10
    conda activate retriever

    # we recommend installing torch with conda for faiss-gpu
    conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    pip install transformers datasets pyserini

    ## install the gpu version faiss to guarantee efficient RL rollout
    conda install -c pytorch -c nvidia faiss-gpu=1.8.0

    ## API function
    pip install uvicorn fastapi

    ```

2. Index 다운로드
    ```bash
    # 출처: https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md
    save_path=downloads
    huggingface-cli download PeterJinGo/wiki-18-bm25-index --repo-type dataset --local-dir $save_path
    ```


2. Retriever launch
    ```bash
    conda activate retriever

    save_path=downloads
    index_file=$save_path/bm25
    corpus_file=$save_path/wiki-18.jsonl
    retriever_name=bm25

    python search_r1/search/retrieval_server.py --index_path $index_file --corpus_path $corpus_file --topk 3 --retriever_name $retriever_name
    ```


## GraphCheck 실행 방법

Retriver 를 실행한 다음, graphcheck_v1/v2/v3.sh 파일을 sbatch로 제출하시면 각 버전의 GraphCheck가 실행됩니다.

```bash
sbatch graphcheck_v1.sh
```

GraphCheck 실행 결과는 `results` 폴더 내에 저장되며,

아래 명령어를 추가로 실행하면 결과가 저장된 경로에 EM/F1 score 가 담긴 .out 파일이 생성됩니다.
```
python utils/agg_eval.py --input_path <result json 파일 경로>
```

## 실험 대상 데이터셋 관련 정보
- `datasets` 폴더 내에 각 데이터셋별 json 파일이 있습니다.
- 각 데이터셋의 `train_sampled.json` 이 저희가 실험으로 사용할 샘플 데이터셋입니다. (train set에서 일부 샘플링)