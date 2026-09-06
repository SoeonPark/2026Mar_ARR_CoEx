# vLLM + LoRA Implementation Audit for CoEx

Audit 기준: 2026-06-24 현재 working tree. 이 문서는 코드와 설치된 dependency(`peft 0.18.0`, 현재 Transformers/vLLM 구현)를 정적으로 추적한 결과다. 학습 코드는 수정하지 않았다. 아래에서 `D`는 diversity adapter 수, `A = 1 + D`는 rollout에 참여하는 전체 adapter 수를 뜻한다.

## 1. Executive Summary

- **current LoRA implementation:** base 4-bit CausalLM에 `get_peft_model()`로 `default` adapter를 만들고, `diversity_0 ... diversity_{D-1}`를 `add_adapter()`로 시작 시 모두 resident 등록한다 (`main.py:189-213`, `main.py:224-246`). 학습 loop 안에서 `add_adapter`, `load_adapter`, `delete_adapter`는 호출되지 않는다.
- **current vLLM integration:** 실제 run script는 모두 colocated vLLM을 사용한다 (`run_coex_0.sh:146-147`, `run_coex_1.sh:158-159`). source별 prompt를 adapter별로 나눠 `A`개의 vLLM generation call로 직렬 실행한다 (`custom_coex_trainer.py:3447-3491`). 각 adapter는 stable name/id/path를 갖지만 `LoRARequest` 객체는 매번 다시 만든다 (`custom_coex_trainer.py:2742-2752`, `custom_coex_trainer.py:2861-2871`).
- **most likely 3.5x bottleneck:** `set_adapter()` 자체보다 다음 네 항목이 훨씬 유력하다.
  1. 10개 completion을 한 번에 continuous-batch하지 않고 4/3/3처럼 source별 `A`번 generation하여 decoding wall time을 거의 합산한다 (`custom_coex_trainer.py:3447-3491`).
  2. sync 시 adapter 하나만 저장한다는 주석과 달리 `save_pretrained(..., selected_adapters=None)`가 **모든 adapter**를 매 path에 저장한다 (`custom_coex_trainer.py:1821-1835`). 따라서 A개 path에 A개 adapter를 쓰는 O(A²) disk serialization이다. 1.5B 체크포인트 실측 adapter 하나는 약 73.9 MB이므로 A=3이면 optimizer step당 약 9 adapter-copy, 약 665 MB를 덮어쓸 수 있다.
  3. `logging_steps=1`인 run script와 결합해 매 optimizer step 모든 LoRA tensor를 fp32 CPU로 복사해 fingerprint를 계산한다 (`custom_coex_trainer.py:923-989`, `custom_coex_trainer.py:5112-5114`, `run_coex_0.sh:167`). A=3 실측 총 55,394,304 parameters, 약 211 MiB의 fp32 D2H materialization이다. `adapter_sanity_check_steps`는 config에만 있고 이 경로를 throttle하지 않는다.
  4. current/reference logprob scoring이 adapter group별로 반복되고, current scoring은 token chunk마다 LM-head forward + entropy forward + backward checkpoint recompute를 수행한다 (`custom_coex_trainer.py:1349-1439`, `custom_coex_trainer.py:1588-1713`, `custom_coex_trainer.py:4887-4903`).
- **additional non-model overhead:** 매 rollout마다 pretty-printed completion JSON 저장 (`custom_coex_trainer.py:2246-2249`), 매 adapter generation마다 두 군데의 decode debug print (`custom_coex_trainer.py:2900-2913`, `custom_coex_trainer.py:3343-3348`), 매 loss call debug print (`custom_coex_trainer.py:4905-4907`)가 켜져 있다. 관찰된 log 파일도 486 MB/861 MB이므로 I/O를 무시할 수 없다.
- **highest-confidence adapter mismatch:** PEFT `save_pretrained()`는 active adapter만 저장하지 않는다. `selected_adapters`가 없으면 모든 adapter를 저장하며 `default`는 지정 path의 root, diversity adapters는 하위 디렉터리에 저장한다. 그런데 vLLM에는 각 adapter의 부모 path를 넘긴다 (`custom_coex_trainer.py:790-803`, `custom_coex_trainer.py:1829-1834`). 설치된 vLLM loader는 전달받은 path 바로 아래의 `adapter_config.json`과 `adapter_model.safetensors`를 읽는다. 따라서 현재 `diversity_i` request도 부모 path root의 **default weights를 읽는 구조**다. name/id는 diversity여도 실제 tensor는 default일 가능성이 아니라, 현재 dependency semantics상 그렇게 읽힌다.
- **reset/load risk:** run scripts에 `--resume_from_checkpoint`가 있으나 (`run_coex_0.sh:201`, `run_coex_1.sh:219`) `main.py`는 `trainer.train()`에 이를 전달하지 않는다 (`main.py:292-293`). 따라서 현재 실행은 checkpoint를 resume하지 않고 새로 초기화한 adapter로 시작한다. 이것이 “중간 reset”처럼 관찰될 수 있다. 향후 단순히 인자를 전달해 resume를 활성화해도, 설치된 Transformers multi-adapter loader는 diversity subdirectories가 있으면 root의 default adapter를 로드하지 않는 분기이므로 별도 load 검증이 필요하다.
- **training-model reset by vLLM sync:** 현재 sync는 training PEFT model에서 파일로 쓰고 vLLM에 읽히는 단방향이다 (`custom_coex_trainer.py:1799-1866`). training tensor를 vLLM에서 역으로 load하는 코드는 없으므로 정상 `save_pretrained()` 자체가 training weights를 초기화할 이유는 없다. 위험은 training model reset보다는 잘못된 adapter export/path, resume 미작동, silent vLLM reload failure/staleness다.

## 2. Adapter Creation and Registration

### 2.1 Creation flow

1. Base model은 NF4 4-bit로 load된다 (`main.py:184-213`).
2. LoRA 상수는 `r=16`, `alpha=32`, `dropout=0.05`다 (`main.py:44-46`).
3. `LoraConfig`는 다음과 같다 (`main.py:224-235`).

| field | current value |
|---|---|
| `r` | 16 |
| `lora_alpha` | 32 |
| `lora_dropout` | 0.05 |
| `target_modules` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| `bias` | 명시하지 않아 PEFT default `none` |
| `task_type` | `CAUSAL_LM` |
| `init_lora_weights` | 명시하지 않아 `True`; A는 일반 초기화, B는 0 초기화 |

실제 checkpoint의 `adapter_config.json`도 `bias=none`, `r=16`, `alpha=32`, `dropout=0.05`, 위 7개 target, `CAUSAL_LM`, `init_lora_weights=true`를 확인했다.

4. `get_peft_model(model, peft_config)`가 이름 `default`를 생성한다 (`main.py:243`).
5. `for i in range(num_diversity_adapters)`에서 `add_adapter(f"diversity_{i}")`를 한 번씩 호출한다 (`main.py:245-247`).
6. `default`를 active로 선택한다 (`main.py:250-253`).
7. Trainer가 사용하는 adapter 목록도 `default`, `diversity_0...` 규칙이다 (`custom_coex_trainer.py:299-304`). 단, diversity completion 수가 0이면 adapters는 PEFT model에는 만들어져도 trainer rollout 목록에는 들어가지 않는다.

### 2.2 Resident/attach behavior

- 모든 adapter는 `CoExTrainer` 생성 전에 동일 PEFT model에 resident 등록된다.
- training loop에는 `add_adapter`, `load_adapter`, `delete_adapter` 호출이 없다. 검색 결과 active code의 `add_adapter`는 `main.py:246` 한 곳뿐이다.
- `set_adapter()`는 attach/load가 아니라 resident module의 active key를 바꾸는 switch다.
- 같은 이름으로 `add_adapter()`를 반복 호출하는 active path는 없다.
- 새 weight가 생성되는 path는 프로세스 시작 시 `get_peft_model()`/`add_adapter()`뿐이다. 정상 training loop의 `set_adapter()`나 `save_pretrained()`는 weight를 재초기화하지 않는다.

### 2.3 `set_adapter()` and `requires_grad`

설치된 PEFT의 `set_adapter()`는 active adapter를 바꾸면서 active adapter parameter만 `requires_grad=True`, 나머지를 `False`로 만든다. CoEx는 이를 알고 현재도 `enable_all_lora_grads()`를 유지한다 (`custom_coex_trainer.py:875-882`). 이 helper는 다음 지점에서 호출된다.

- Trainer base initialization 직후: `custom_coex_trainer.py:588`
- optimizer 생성 직전: `custom_coex_trainer.py:905-908`
- 매 adapter training forward 직전: `custom_coex_trainer.py:1129-1130`
- vLLM export switch 직후: `custom_coex_trainer.py:1821-1822`
- policy-repulsion scoring switch 직후: `custom_coex_trainer.py:2306-2307`
- correctness scoring switch 직후: `custom_coex_trainer.py:3694-3695`

따라서 과거 callback이 제거된 상태가 아니라, callback 대신 trainer method로 **현재 active하게 존재**한다. 백업 파일에도 같은 helper가 있어 최근에 새로 생긴 임시 코드도 아니다.

주의할 점은 vLLM init adapter export loop (`custom_coex_trainer.py:789-803`)가 마지막 diversity adapter를 active로 남기고 그 뒤 `enable_all_lora_grads()`를 다시 호출하지 않는다는 것이다. 그러나 실제 optimizer 생성 직전 `create_optimizer()`가 다시 전부 enable하므로 현재 기본 Trainer lifecycle에서는 optimizer 누락으로 이어지지 않는다.

### 2.4 Parameter counts and optimizer membership

현재 1.5B checkpoint를 직접 읽은 결과 adapter별 safetensors는 392 tensor, 18,464,768 parameters다.

| adapter | num_params | requires_grad at optimizer creation | in optimizer |
|---|---:|---:|---:|
| `default` | 18,464,768 | 18,464,768 | 18,464,768 |
| `diversity_0` | 18,464,768 | 18,464,768 | 18,464,768 |
| `diversity_1` | 18,464,768 | 18,464,768 | 18,464,768 |

근거:

- adapters는 `CoExTrainer`/optimizer보다 먼저 생성된다 (`main.py:243-277`).
- `create_optimizer()`가 모든 `lora_` parameter를 enable한 다음 parent optimizer를 만든다 (`custom_coex_trainer.py:905-908`).
- 생성 직후 `_assert_all_lora_in_optimizer()`가 LoRA parameter id 전체가 optimizer param groups에 있는지 검사하고 하나라도 빠지면 예외를 낸다 (`custom_coex_trainer.py:884-902`).
- 저장된 정상 optimizer를 읽은 결과 group 0에 1,176 tensors (=392×3), group 1은 0 tensors였다. 세 adapter 모두 group 0에 들어간 상태와 일치한다.

`set_adapter()` 직후에는 inactive adapters가 잠시 freeze되지만, CoEx의 바로 다음 `enable_all_lora_grads()`가 되돌린다. forward에는 active adapter만 참여하므로 inactive adapter는 `requires_grad=True`여도 그 forward에서 gradient가 생기지 않으며, adapter별 training forward가 차례로 실행되면서 각 adapter gradient가 같은 optimizer step에 누적된다 (`custom_coex_trainer.py:1127-1214`).

권장 runtime membership sanity check:

```python
for name, p in model.named_parameters():
    if "lora" in name.lower():
        print(name, p.requires_grad, p.data.norm().item(), id(p))

opt_param_ids = {
    id(p) for group in trainer.optimizer.param_groups for p in group["params"]
}
for name, p in model.named_parameters():
    if "lora" in name.lower():
        print("[OPT]", name, id(p) in opt_param_ids)
```

권장 group summary는 adapter name을 parameter name에서 파싱해 `group_id / tensor_count / numel / adapter별 numel`을 출력하는 방식이다. 현재 `_assert_all_lora_in_optimizer()`는 pass/fail만 제공하고 group별 숫자는 출력하지 않는다.

### 2.5 Save/load lifecycle and reset paths

- Trainer checkpoint save는 `_save_checkpoint()`에서 parameter 전체를 print한 뒤 parent save를 호출한다 (`custom_coex_trainer.py:5257-5267`). PEFT `save_pretrained()`의 default 동작으로 root에 `default`, 하위 `diversity_i/`에 각 adapter가 저장되는 것을 실제 checkpoint에서 확인했다.
- `main.py`의 final save도 adapter를 `set_adapter()`한 뒤 `save_pretrained()`하지만 `selected_adapters`를 지정하지 않는다 (`main.py:295-309`). 따라서 `final_checkpoint_diversity_i`라는 이름과 달리 각 디렉터리에도 모든 adapters가 저장된다.
- `--resume_from_checkpoint`는 현재 `trainer.train()`에 전달되지 않아 **실행되지 않는다** (`main.py:292-293`). 현재 resume run script는 이름만 resume이며 model/optimizer/global step을 복원하지 않는다.
- 향후 resume를 켤 때는 root default와 모든 subdir weights를 명시적으로 load하고, load 전후 adapter별 hash/norm 비교가 필요하다. 설치된 Transformers loader의 multi-adapter 분기는 subdirectories를 순회할 때 root default를 별도로 읽지 않는다.

## 3. Adapter Switching and Generation Flow

### 3.1 Source allocation and preservation

- `main.py:387-392`가 `num_generations = main_count + D*diversity_count`로 정하고 generation/per-device batch를 같은 값으로 덮어쓴다.
- `_prepare_inputs()`는 flat generation axis를 adapter 구간으로 나눠 `adapter_to_indices`를 만든다 (`custom_coex_trainer.py:1961-1982`). 예: `default=[0..3]`, `diversity_0=[4..6]`, `diversity_1=[7..9]`.
- `_generate_completions()`가 이 map을 보존해 source별 생성 후 원래 flat 위치로 되돌린다 (`custom_coex_trainer.py:3447-3505`).
- 이후 `default` view는 의도적으로 전체 flat batch를 받는다 (`custom_coex_trainer.py:3626-3631`). diversity view는 자기 indices만 받는다. 즉 main correctness/update는 모든 source의 completions를 보지만, diversity update는 자기 source만 본다.
- explicit `source_id` tensor는 최종 training batch에 저장되지 않는다. source 정보는 `adapter_to_indices`, 분리된 per-adapter dict, diversity comparison용 `adapter_names`로만 간접 보존된다 (`custom_coex_trainer.py:2046-2060`). sample-wise routed MultiLoRA에는 explicit source-id tensor 추가가 필요하다.

### 3.2 Current script 기준 one optimizer step 호출 수

현재 대표 scripts는 `gradient_accumulation_steps=2` (`run_coex_0.sh:152`), `generation_batch_size == per_device_train_batch_size`여서 config가 `steps_per_generation=1`로 계산한다 (`custom_coex_config.py:896-908`). `num_iterations=1` default이므로 `_prepare_inputs()`는 **각 training microstep마다** generation한다 (`custom_coex_trainer.py:1961-1962`). 따라서 optimizer step 하나에 rollout generation event가 2번이다.

`_last_loaded_step_per_adapter`는 `global_step` 기준이라 (`custom_coex_trainer.py:561`, `custom_coex_trainer.py:2732-2739`) 같은 optimizer step의 첫 microstep에서만 vLLM sync하고 두 번째 microstep은 같은 weights를 재사용한다.

trace-Jaccard / BLEU / external reward, `use_importance_weighting=False`인 경우:

| phase | per optimizer step | `set_adapter()` calls |
|---|---:|---:|
| generation event | 2 | sync가 있는 첫 event에서 A |
| vLLM generation calls | 2A | 0 additional; vLLM request로 route |
| correctness/old/ref preparation | 2A source scoring calls | 2A |
| diversity reward | 2D | 0 for trace/BLEU/external |
| current logprob + backward | 2A adapter batches | 2A |
| **total** |  | **5A** |

따라서 4/3/3, D=2, A=3 run은 optimizer step당 약 **15회**다. 4/2/2/2, D=3, A=4 short run은 약 **20회**다. 이것은 `disable_adapter()` 내부 enable/disable와 initialization/final save를 제외한 explicit CoEx `set_adapter()` 수다.

policy-repulsion `target=all_other`이면 각 diversity source마다 source 1회 + other A-1회 scoring한다 (`custom_coex_trainer.py:2396-2449`). optimizer step당 추가 호출은 `2*D*A`이고, D=2/A=3이면 12회가 추가되어 총 약 27회다. 이 경우 `policy_repulsion_batch_size=1` default도 forward 직렬화를 크게 만든다 (`custom_coex_config.py:799-802`).

### 3.3 Old/current/reference logprob adapter

- **old logprob:** 주석으로 남은 “vLLM이면 항상 계산” 경로는 비활성화되어 있다 (`custom_coex_trainer.py:3697-3743`). active code는 `use_importance_weighting=True`, adapter=`default`, vLLM sampling logprobs가 있을 때만 current training model의 `default`로 계산한다 (`custom_coex_trainer.py:3745-3768`). 현재 scripts는 False라 old forward는 없다.
- **reference logprob:** scripts의 `beta=0.04` (`run_coex_0.sh:169`)이므로 계산한다. PEFT model에서는 별도 ref model이 없고 active adapter를 `disable_adapter()`하여 공통 base policy로 scoring한다 (`custom_coex_trainer.py:3783-3807`). 어느 source든 reference adapter는 “none/base”다.
- **current logprob:** `training_step()`이 per-adapter dict를 순회하며 해당 adapter를 set한 뒤 (`custom_coex_trainer.py:1127-1131`), `_compute_loss()`에서 gradient가 연결된 selected logprobs를 계산한다 (`custom_coex_trainer.py:4876-4903`).
- **backward 종료 active adapter:** loop의 마지막 adapter, 보통 `diversity_{D-1}`다. 다만 모든 LoRA grad는 helper로 enable된 상태다.
- **reward:** trace/BLEU/correctness reward 자체는 adapter switch가 없다. policy-repulsion만 source 및 comparison adapters로 반복 switch한다 (`custom_coex_trainer.py:4007-4016`, `custom_coex_trainer.py:2430-2449`).

### 3.4 Suggested switch tracer

PEFT wrapper와 호출 phase를 함께 기록해야 한다. 단순 monkey patch는 다음처럼 시작할 수 있다.

```python
orig_set_adapter = trainer.model.set_adapter
trainer._adapter_trace_phase = "unknown"
trainer._adapter_trace_count = 0

def traced_set_adapter(name):
    trainer._adapter_trace_count += 1
    before = getattr(trainer.model, "active_adapter", None)
    t0 = time.perf_counter()
    out = orig_set_adapter(name)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(
        f"[SET_ADAPTER] step={trainer.state.global_step} "
        f"micro={trainer._step} phase={trainer._adapter_trace_phase} "
        f"before={before} after={name} seconds={dt:.6f}"
    )
    return out

trainer.model.set_adapter = traced_set_adapter
```

`generation_sync`, `correctness_old_ref`, `diversity_reward`, `current_backward` 진입 시 phase tag를 바꾸면 optimizer step별 breakdown을 얻는다. `before == name`도 별도 count하여 불필요한 no-op switch를 찾는 것이 좋다.

## 4. vLLM LoRARequest Flow

### 4.1 Engine creation

- server mode는 `VLLMClient`만 만든다 (`custom_coex_trainer.py:715-723`). 그러나 `lora_modules` 초기화/export는 colocate 분기에만 있어 현재 PEFT server mode는 완성된 multi-adapter 경로가 아니다.
- colocate mode는 base model path로 `LLM(...)`을 만들고 `enable_lora=True`, `max_lora_rank=args.lora_r` 또는 64로 설정한다 (`custom_coex_trainer.py:755-779`). 현재 `CoExConfig`에 `lora_r` field가 없으므로 vLLM limit은 64, 실제 rank 16은 허용 범위다.
- 각 process가 `tempfile.mkdtemp(prefix="vllm_lora_cache_")`로 temp root를 만든다 (`custom_coex_trainer.py:784`).

### 4.2 Name/id/path table

`custom_coex_trainer.py:788-803`에서 다음 mapping을 고정한다.

| source_id | adapter_name | lora_request_id | path | object reuse | intended sync |
|---:|---|---:|---|---|---|
| 0 | `default` | 1 | `<tmp>/default_adapter` | object recreated | once per global step |
| 1 | `diversity_0` | 2 | `<tmp>/diversity_0_adapter` | object recreated | once per global step |
| i+1 | `diversity_i` | i+2 | `<tmp>/diversity_i_adapter` | object recreated | once per global step |

name/id/path collision은 CoEx mapping 자체에는 없다. IDs는 process 내 adapter order에 대해 stable하다. 다만 **path content가 collision-equivalent**다. 모든 parent path의 root가 default adapter이기 때문이다.

### 4.3 Sync and cache behavior

1. `_generate_single_turn()`이 adapter/global-step pair가 아직 sync되지 않았으면 `_move_model_to_vllm(adapter_name)`을 호출한다 (`custom_coex_trainer.py:2732-2739`).
2. `_move_model_to_vllm()`은 training model을 adapter로 switch하고 temp path에 save한다 (`custom_coex_trainer.py:1809-1835`).
3. colocate engine에서 동일 numeric id를 `remove_lora()`한 뒤 새 `LoRARequest`로 `add_lora()`한다 (`custom_coex_trainer.py:1763-1788`).
4. prefix cache를 reset한다 (`custom_coex_trainer.py:1791-1795`, `custom_coex_trainer.py:1863-1866`).
5. generation 직전에도 `LoRARequest`를 두 번 생성한다. branch 공통 위치 (`custom_coex_trainer.py:2740-2752`)에서 한 번, colocate 분기 (`custom_coex_trainer.py:2861-2871`)에서 다시 만들어 앞 객체를 덮어쓴다.
6. `clear_KV_cache_after_generation=True`이면 source별 generation 뒤 다시 prefix cache를 reset한다 (`custom_coex_trainer.py:2941-2947`, `run_coex_0.sh:198`). adapter별 cache reuse는 사실상 거의 없다.

현재 대표 A=3, grad accumulation=2 기준 optimizer step당:

- actual vLLM generation requests: `2A = 6`
- engine remove/add reload requests: `A = 3`
- constructed `LoRARequest` objects: 첫 event `3A`(reload+중복 2개), 둘째 event `2A`, 합계 `5A = 15`
- adapter file sync/export: A회 호출이지만 각 호출이 모든 A adapters를 저장하므로 A² adapter files

### 4.4 Integrity risk matrix

| risk | assessment | evidence |
|---|---|---|
| training LoRA updated, vLLM stale | single-process 정상 경로에서는 다음 global step 첫 generation 전에 sync. reload failure가 warning으로만 처리되거나 multi-process면 stale 가능 | `custom_coex_trainer.py:2732-2739`, `custom_coex_trainer.py:1771-1795` |
| export initializes training weights | 정상 PEFT save는 read-only라 직접 reset 가능성 낮음 | `custom_coex_trainer.py:1829-1834` |
| wrong adapter exported | **확정적인 구조 문제.** `selected_adapters` 미지정으로 parent root는 항상 default | `custom_coex_trainer.py:790-803`, `custom_coex_trainer.py:1829-1834` |
| name/id cache reuse anomaly | same id를 명시적으로 remove/add하므로 의도는 reload. 예외를 삼키므로 실패 시 상태가 불명확 | `custom_coex_trainer.py:1771-1789` |
| training active vs request mismatch | training model switch name과 request metadata name은 일치하지만 path root tensor가 불일치 | `custom_coex_trainer.py:1821`, `custom_coex_trainer.py:1781-1785` |
| multi-process stale temp paths | main process만 sync save하고 각 rank는 서로 다른 temp dir을 가질 수 있음. non-main path는 init snapshot에 머물 수 있음 | `custom_coex_trainer.py:784`, `custom_coex_trainer.py:1827-1835` |
| server mode PEFT sync | `lora_modules`가 colocate에서만 정의되고 remote adapter update가 구현되지 않음 | `custom_coex_trainer.py:715-724`, `custom_coex_trainer.py:784-805` |

## 5. LoRA Weight Integrity Checks

### 5.1 Existing checks

- `_lora_fingerprint()`은 adapter별 sum/sumsq/max/subsample/numel을 계산한다 (`custom_coex_trainer.py:923-955`).
- `_log_lora_fingerprints()`는 rank 평균과 이전 fingerprint delta를 기록한다 (`custom_coex_trainer.py:980-1005`).
- 이것은 `log()`마다 무조건 실행된다 (`custom_coex_trainer.py:5112-5114`). `adapter_sanity_check_steps` (`custom_coex_config.py:314-317`)는 실제로 사용되지 않는다.
- `_assert_all_lora_in_optimizer()`는 membership을 보장한다 (`custom_coex_trainer.py:884-908`).
- chunked scorer에는 wrapper/base active adapter 일치 및 full-vs-chunk selected logprob 비교가 있다 (`custom_coex_trainer.py:1443-1584`).

기존 fingerprint는 A/B norm 분리, optimizer membership 수, active adapter, grad norm, vLLM file hash, save/load round trip을 기록하지 않으므로 reset/stale 진단에는 불충분하다.

### 5.2 Proposed summary helper

```python
@torch.no_grad()
def summarize_lora(trainer, tag):
    model = trainer.accelerator.unwrap_model(trainer.model)
    opt_ids = set()
    if trainer.optimizer is not None:
        opt_ids = {
            id(p)
            for group in trainer.optimizer.param_groups
            for p in group["params"]
        }
    active = getattr(model, "active_adapter", None)
    print(f"\n[LORA_SUMMARY] tag={tag} step={trainer.state.global_step} active={active}")
    for adapter in trainer.all_adapter_names:
        a_norm = b_norm = grad_norm = 0.0
        numel = trainable = in_optimizer = grad_numel = 0
        for name, p in model.named_parameters():
            if "lora_" not in name or not trainer._match_adapter_param(name, adapter):
                continue
            numel += p.numel()
            trainable += p.numel() if p.requires_grad else 0
            in_optimizer += p.numel() if id(p) in opt_ids else 0
            if "lora_A" in name:
                a_norm += p.detach().float().norm().item()
            elif "lora_B" in name:
                b_norm += p.detach().float().norm().item()
            if p.grad is not None:
                grad_norm += p.grad.detach().float().norm().item()
                grad_numel += p.numel()
        print({
            "adapter": adapter,
            "A_norm_sum": a_norm,
            "B_norm_sum": b_norm,
            "numel": numel,
            "trainable": trainable,
            "in_optimizer": in_optimizer,
            "grad_numel": grad_numel,
            "grad_norm_sum": grad_norm,
            "active": active,
        })
```

전체 tensor를 `.cpu()`로 복사하지 말고 scalar norm만 `.item()`해야 profiling 자체가 병목이 되지 않는다. reset 여부를 강하게 확인하려면 각 adapter에서 고정된 2~4개 tensor의 SHA256 또는 deterministic sampled checksum도 함께 기록한다.

### 5.3 Required checkpoints

다음 tag에서 위 summary와 sampled hash를 기록한다.

1. `after_add_adapter` — B norm이 0인 정상 초기화 baseline 확보
2. `after_optimizer_create` — 모든 adapter `in_optimizer == numel`
3. `before_rollout_generation`
4. `after_rollout_generation` — training tensor hash가 3과 동일해야 함
5. `before_vllm_sync`
6. `after_vllm_sync` — training tensor hash가 5와 동일해야 함
7. `before_logprob_scoring`
8. `after_backward` — source adapter에 nonzero grad가 있는지 확인
9. `after_optimizer_step` — 학습된 adapter hash/norm delta 확인
10. `before_checkpoint_save`
11. `after_checkpoint_load` — memory tensor와 saved tensor를 adapter별 exact/allclose 비교

vLLM sync는 training norm만 비교해서는 부족하다. 각 `adapter_info["path"]`의 **root safetensors key/value hash**와 training adapter의 PEFT state dict hash를 비교해야 한다. 현재 bug는 training norm이 정상인데 path root가 다른 adapter인 경우이므로 다음 invariant가 핵심이다.

```text
request.name == expected adapter name
request.id == stable adapter id
exported directory contains exactly one selected adapter at its root
hash(training PEFT state for adapter) == hash(exported root state)
hash(exported root before add_lora) == hash(vLLM-loaded adapter, if introspection available)
training hash(before sync) == training hash(after sync)
```

판정:

- B norm이 학습 중 0/초기 checksum으로 되돌아감: reset 의심
- optimizer step 후에도 A/B/hash/grad가 계속 불변: optimizer 누락, no-grad, zero advantage 확인
- sync 후 training hash 변화: 심각한 역방향 mutation bug
- training hash와 exported root hash 불일치: 현재처럼 wrong-adapter export
- save/load hash 불일치: checkpoint/resume bug

## 6. Runtime Profiling Plan

### 6.1 Existing profiling gaps

`@profiling_decorator`와 `profiling_context("vLLM.generate")`는 일부 함수에 있지만 (`custom_coex_trainer.py:1587`, `custom_coex_trainer.py:1798`, `custom_coex_trainer.py:2779`, `custom_coex_trainer.py:2890`) 요청한 step-level component table은 만들지 않는다. `memory_profiling`, `memory_profile_interval` flags는 config/run scripts에만 있고 trainer에서 읽히지 않는다. `torch.cuda.max_memory_allocated()`도 현재 code에 없다.

### 6.2 Timing points

CPU/I/O 구간은 `time.perf_counter()`, CUDA forward/backward는 CUDA events 또는 양쪽 `torch.cuda.synchronize()`를 사용한다. 모든 sync를 production run에 남기지 말고 profile mode에만 켠다.

| metric | start/end location |
|---|---|
| `generation_main_time` | `_generate_completions()`의 default `_generate()` 전후, `custom_coex_trainer.py:3447-3491` |
| `generation_diversity_time` | 같은 loop의 diversity calls 합 |
| `vllm_lora_request_time` | request construction + `eng.remove_lora/add_lora`, `custom_coex_trainer.py:1763-1789`, `2742-2752`, `2861-2871` |
| `vllm_weight_sync_time` | `_move_model_to_vllm()`, export와 reload를 별도 sub-timer, `custom_coex_trainer.py:1799-1866` |
| `adapter_set_time` | monkey-patched `set_adapter()` 누적 |
| `old_logprob_scoring_time` | `custom_coex_trainer.py:3745-3760` |
| `reference_logprob_scoring_time` | `custom_coex_trainer.py:3783-3807` |
| `current_logprob_scoring_time` | `_compute_loss()`의 call, `custom_coex_trainer.py:4887-4903` |
| `diversity_reward_scoring_time` | `_score_completions_diversity()`, policy-repulsion forward와 CPU reward 분리, `custom_coex_trainer.py:3985-4095` |
| `reward_function_time` | `_calculate_rewards`/`_calculate_diversity_rewards`, `custom_coex_trainer.py:2500-2624` |
| `backward_time` | 각 `accelerator.backward`, `custom_coex_trainer.py:1190-1212` |
| `optimizer_step_time` | Trainer loop의 optimizer step 전후 callback/wrapper; `training_step()` 밖임 |
| `fingerprint_time` | `custom_coex_trainer.py:5114` |
| `completion_json_time` | `custom_coex_trainer.py:2246-2249` |
| `total_step_time` | optimizer-step boundary callback 전후 |

매 optimizer step 출력 schema:

```text
step num_sources num_total_completions
num_set_adapter_calls num_vllm_generate_calls num_vllm_lora_objects num_vllm_reloads
generation_main_time generation_diversity_time
vllm_weight_export_time vllm_reload_time adapter_set_time
old_logprob_time current_logprob_time reference_logprob_time
diversity_reward_scoring_time reward_function_time
backward_time optimizer_step_time fingerprint_time json_io_time total_time
cuda_peak_allocated cuda_peak_reserved
```

step 시작에 `torch.cuda.reset_peak_memory_stats()`, 끝에 allocated/reserved peak를 기록한다. 각 adapter마다 `num_prompts`, generated tokens, scoring tokens도 기록해야 시간 차이를 token 수로 normalize할 수 있다.

### 6.3 Ablation order and decision criteria

동일 prompt/completion length seed로 다음 profile을 최소 20 optimizer steps 수행한다.

1. instrumentation only, current code
2. debug decode print / completion JSON / fingerprint를 profile에서만 off
3. vLLM export와 `remove/add` 시간을 별도 측정하되 generation 동일
4. source generation을 한 vLLM call로 합친 prototype
5. `beta=0` profile로 reference scoring 제거
6. `logprob_token_chunk_size` 64/128/256 sweep
7. policy-repulsion이면 별도 off/on 및 batch size sweep

판정:

- `adapter_set_time / total < 1%`: switching 자체는 원인이 아님
- generation 세 source 합이 baseline single call 대비 약 3x: source별 serial generation이 주원인
- export/reload가 크고 disk write bytes가 A²: LoRA sync implementation이 주원인
- fingerprint/JSON/debug off에서 큰 개선: instrumentation I/O가 주원인
- current/ref scoring이 큼: B-lite 가치가 큼
- chunk size 증가로 time은 줄고 peak memory만 증가: [B,chunk,V] projection/checkpoint overhead가 원인

현재 구현은 full `[B,T,V]` logits를 보존하지 않는다. `[B,chunk_T,V]`만 만들고 즉시 selected logps로 줄인다 (`custom_coex_trainer.py:1330-1439`). 다만 current scoring은 entropy를 위해 같은 chunk를 별도로 projection하고 backward 때 checkpoint recompute하므로 LM-head 계산은 최대 3회다. 문제는 full-logit copy라기보다 chunk 반복, entropy duplicate, checkpoint recompute다.

## 7. Current Implementation Diagrams

### 7.1 Training step flow

```text
prompt batch
  │ source_id: flat position only; adapter_to_indices builds main/div groups
  │ weights: read=no, write=no, grad=no
  ▼
source allocation (custom_coex_trainer.py:1966-1982)
  │ source_id preserved in index map; no explicit source_id tensor
  ▼
for adapter in [default, diversity_0, ...]             [SERIAL]
  ├─ first microstep/global_step:
  │    training PEFT set_adapter(adapter)               active=adapter
  │    save_pretrained(temp/<adapter>_adapter)          weights read; no intended write
  │    vLLM remove(id) -> add(LoRARequest)              no gradient
  │    WARNING: exported root currently contains default weights
  └─ vLLM chat(..., one LoRARequest)                    weights read in vLLM; no gradient
       completion returned to original flat indices
  ▼
completion collection (3386-3658)
  │ default view = all sources; diversity view = own source
  ▼
correctness reward
  │ CPU/python or reward model; no policy gradient
  │ set_adapter(source) occurs even when old logprob is disabled
  ▼
reference scoring (beta != 0)
  │ active adapter disabled => base policy
  │ weights read; no write; torch.no_grad
  ▼
diversity reward
  ├─ trace/BLEU/external: no adapter switch
  └─ policy repulsion: source + comparison adapter forwards [SERIAL]
       weights read; no grad
  ▼
advantage computation
  │ source grouping preserved by per-adapter dict
  ▼
for adapter in [default, diversity_0, ...]             [SERIAL]
  ├─ set_adapter(adapter); force all LoRA requires_grad=True
  ├─ current selected-logprob scoring
  │    active adapter weights read; autograd graph built
  ├─ loss
  └─ backward
       active adapter LoRA gradients written/accumulated
  ▼
optimizer step (outside CoEx training_step, after grad accumulation)
  │ all adapters in optimizer; LoRA weights written
  ▼
next global_step first generation
  └─ vLLM sync/export happens lazily before generation
```

중요한 순서 차이: 요청 예시의 “optimizer step -> vLLM sync”는 실제로는 optimizer 직후 즉시 callback이 아니라 **다음 rollout의 첫 `_generate_single_turn()` 직전 lazy sync**다 (`custom_coex_trainer.py:2732-2739`).

### 7.2 Adapter lifecycle

```text
LoraConfig
  -> get_peft_model(default)                            [new A/B initialization]
  -> add_adapter(diversity_i) once each                 [new A/B initialization]
  -> all resident in one PEFT model
  -> CoExTrainer init
  -> vLLM temp export loop                              [RISK: saves all adapters/path]
  -> create_optimizer: force all grads, assert IDs      [optimizer registration OK]
  -> generation: PEFT export -> vLLM remove/add         [RISK: root default mismatch]
  -> scoring: set active adapter                        [inactive grad flags temporarily change]
  -> backward/optimizer update                          [weights written]
  -> next-generation lazy sync                          [RISK: silent reload failure/multiprocess stale]
  -> Trainer checkpoint save                            [all adapters saved: root + subdirs]
  -> optional reload                                    [CURRENTLY NOT INVOKED]
       if enabled: root default + every subdir must be verified
  -> final save                                         [RISK: path name says one adapter, all are saved]
```

Reset/stale/mismatch flags:

- **reset:** only initial creation or an incorrectly implemented future load should initialize; no add/load in active loop.
- **stale:** vLLM reload failure is warning-only; multi-process temp dirs can diverge.
- **mismatch:** current vLLM parent path root points to default for every request.
- **resume loss:** current CLI resume option is ignored; enabling stock multi-adapter load still needs default-root verification.

## 8. MultiLoRA / Source-Routed LoRA Feasibility

### 8.1 Answers

1. **PEFT `set_adapter()`를 sample-wise source routing으로 바꿀 수 있는가?** 가능하지만 stock `set_adapter()`만으로는 불가능하다. LoRA layer가 batch row별 adapter id를 받아 해당 A/B projection을 적용하도록 router/context를 추가하거나, 여러 adapter projection을 계산한 뒤 row mask로 combine해야 한다. optimizer에는 이미 모든 adapter가 들어가므로 parameter ownership은 재사용할 수 있다.
2. **generation B-full에 vLLM을 계속 쓸 수 있는가?** 가능하다. 설치된 vLLM `LLM.generate()`는 prompt별 `list[LoRARequest]`를 지원한다. 현재 conversational path의 `LLM.chat()` signature는 단일 request이므로 chat template을 먼저 text/token prompt로 변환하고 `LLM.generate(all_prompts, lora_request=request_per_prompt)`를 호출해야 한다. Transformers generation으로 갈 필요는 없고, 오히려 custom routed KV-cache 구현 부담이 커진다.
3. **scoring/update만 MultiLoRA인 B-lite?** 가능하다. `_get_per_token_logps_and_entropies()`의 backbone/LoRA forward에 `source_id` routing을 넣고 `_compute_loss`를 하나의 mixed batch로 호출하도록 바꾼다. reference base scoring은 adapter-independent이므로 전체 unique completion batch에서 한 번만 계산할 수 있다.
4. **B-lite가 줄이는 것:** current scoring의 adapter별 backbone calls, reference 중복 scoring, training `set_adapter`, 일부 mini-batch overhead, policy-repulsion comparison batching. **못 줄이는 것:** source별 vLLM generation A calls, O(A²) export bug, vLLM reload/cache reset, CPU reward/JSON/fingerprint.
5. **B-full이 줄이는 것:** B-lite 항목 + source별 serial vLLM calls를 1회 mixed-request continuous batch로 합침. decoding wall time과 engine scheduling overhead를 가장 크게 줄일 가능성이 있다.
6. **가장 적은 수정으로 switch 감소:** current `use_importance_weighting=False`에서는 correctness scoring의 `set_adapter(adapter_name)` 뒤 policy forward가 없고 reference는 adapter-disabled base이므로 그 switch는 불필요하다 (`custom_coex_trainer.py:3694-3807`). 먼저 active-name guard로 no-op switch를 제거하고 이 phase switch를 조건부로 만들 수 있다. 단, 이것은 큰 병목인 generation/export를 해결하지 않는다.
7. **reset 방지 invariant:** adapter별 exact selected export, training/export hash equality, sync 전후 training hash equality, request name/id/path/content consistency, optimizer membership, load round-trip hash를 강제해야 한다.

### 8.2 Feasibility summary

```text
[Feasibility]
B-lite:
  possible: yes
  required files:
    custom_coex_trainer.py
    (router config를 노출할 경우 custom_coex_config.py)
  required changes:
    explicit source_id tensor 보존
    sample-wise routed LoRA layer/context
    mixed current-logprob/loss batch
    base reference scoring deduplication
    policy-repulsion adapter×sample expansion의 batched routing
  expected speed impact:
    scoring 비중이 클수록 중간~큼; generation/export 병목은 그대로
  risk:
    PEFT/quantized linear/gradient-checkpoint 호환성
    source별 loss normalization 및 adapter별 gradient 격리 검증 필요

B-full:
  possible: yes, vLLM 유지 가능
  required files:
    custom_coex_trainer.py
    custom_coex_config.py (feature flag/profile controls)
    main.py (startup/export/load invariants)
  required changes:
    adapter별 exact single-adapter export
    모든 updated adapters를 generation 전에 preload/reload
    conversational prompt를 render 후 LLM.generate 사용
    prompt별 list[LoRARequest] 전달
    output/source index remapping 유지
    B-lite routed scorer/update
  expected speed impact:
    source별 serial decoding 제거로 가장 큼; 현재 3.5x slowdown의 핵심 후보를 직접 겨냥
  risk:
    vLLM version별 mixed-LoRA batching/cache behavior
    request-path integrity, adapter capacity/eviction, TP rank sync
    한 adapter reload 실패 시 mixed batch 전체의 silent mismatch
```

## 9. Recommended Next Steps

### 9.1 Minimal instrumentation patch (next change, not applied)

우선 기능 변경 없이 다음만 넣는 작은 patch가 적절하다.

1. phase-aware `set_adapter` count/time
2. vLLM request construction, export, remove/add, generate separate timing/count
3. old/current/ref/reward/backward/optimizer CUDA timing
4. scalar-only adapter A/B/grad norm 및 optimizer membership
5. training state ↔ exported root state sampled hash assertion
6. step-level CUDA peak allocated/reserved
7. fingerprint/JSON/debug-print time도 별도 계측하여 관측 비용을 드러냄

첫 profile에서는 코드를 고치지 말고 현재 behavior 그대로 측정해야 한다. 다만 current export-root mismatch는 generation 의미 자체를 훼손하므로 timing 결과와 별도로 즉시 fail-fast invariant로 확인할 가치가 있다.

### 9.2 Profiling run command

기존 short run을 기반으로 20-step profile을 권장한다.

```bash
CUDA_VISIBLE_DEVICES=0 \
COEX_PROFILE=1 \
COEX_PROFILE_STEPS=20 \
COEX_LOGPROB_SANITY_CHECK=0 \
./run_coex_0_short.sh
```

instrumentation patch에서는 profile table을 JSONL/CSV 한 줄 per optimizer step으로 별도 저장하고 stdout에는 요약만 출력해야 한다. 현재처럼 decoded completions와 모든 parameter를 stdout에 쓰면 timing을 오염시킨다.

### 9.3 Decision criteria

1. export/reload + fingerprint + JSON/debug I/O가 total의 20% 이상이면 먼저 plumbing을 고친다.
2. source별 generation 합이 single mixed generation 예상치의 1.5배 이상이면 B-full vLLM mixed request를 우선한다.
3. current+reference scoring이 total의 30% 이상이면 B-lite를 병행한다.
4. `set_adapter_time`이 5% 미만이면 routing 설계의 목적을 “switch 제거”가 아니라 “forward batching/중복 제거”로 명확히 한다.
5. 어떤 speed 작업보다 먼저 다음 무결성 조건을 통과해야 한다.

```text
all adapter params in optimizer
all source adapters change after optimizer steps with nonzero advantages
training hash unchanged by vLLM sync
request adapter hash == exported root hash == intended training adapter hash
checkpoint save/load hash exact match for default and every diversity adapter
resume starts at requested global_step with restored optimizer state
```

## 10. Bottom Line

현재 CoEx는 “resident PEFT multi-adapter + active switch” 자체는 합리적으로 구성되어 있고 optimizer 등록도 방어적으로 검증한다. 그러나 vLLM bridge는 active adapter만 export한다는 잘못된 가정 때문에 모든 request가 root default weights를 읽을 수 있으며, sync가 O(A²) disk write가 되어 있다. 동시에 rollout은 source별 직렬이고, 매-step fingerprint/JSON/debug I/O와 반복 current/reference scoring도 크다.

따라서 3.5x slowdown을 `set_adapter()` 비용으로 설명할 근거는 약하다. 가장 먼저 검증할 순서는 **(1) vLLM request별 실제 weight hash, (2) export/reload 시간과 write bytes, (3) source별 generation wall time, (4) fingerprint/JSON/debug I/O, (5) current/reference scorer**다. 무결성을 확보한 뒤에는 vLLM의 prompt별 `LoRARequest` list를 이용한 B-full이 가장 큰 속도 개선 후보이고, B-lite는 scoring 비중이 큰 경우의 보완책이다.
