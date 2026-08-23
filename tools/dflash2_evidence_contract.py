"""Pinned identities for the historical Q6_K-only DFlash2 capture."""

SCHEMA = "a3s.power.dflash2-evidence.v1"
TARGET_MODE = "q6-target-only"
CANDIDATE_MODE = "q6-dflash2"
EXPECTED_SOURCE = {
    "power_commit": "32bc4ea54bc2889e7ada584b4b7ad04616e703f6",
    "llama_cpp_commit": "1deefcca395743049c3820ab8f9b15043f3e9446",
    "llama_server_sha256": "d4fcedab36dc30795c77ea1990c2d1496d27c15d99797a24af54ee5c2e792910",
    "performance_report_sha256": "5f0c66d6a9669fbd85fae64e9d7c43c3217195966dc5a9cd8a7b97bb596da689",
    "quality_environment_sha256": "608a7f761e9b4575be7ce2c0c3c49cc9f609954b0474a6aec0f69fefc49ac615",
    "quality_aggregate_sha256": "becc158ec023e739d01946084ca8ba4b1863a3c5fdad89228eafbdbc5a802b42",
}
EXPECTED_REPORT_HASHES = {
    "r01-o1-q6-target-only.json": "f7b71155d8ea46549e7e9a95b89682accf71f093ef1271d05d6448c903b4135a",
    "r01-o2-q6-dflash2.json": "07481c58b276980fd786e9c2e21cbc3f9ddf34f44709e23858e580c8f5e0746f",
    "r02-o1-q6-dflash2.json": "67dafe5b6518ec18eb6e9a0d924788ba2ea22786e7d78f2406c4f0500fda9fcb",
    "r02-o2-q6-target-only.json": "5e85497158fb720f2485aa32d20fe773e9723fbf6bd7e4acea79dba9d0b600d9",
    "r03-o1-q6-target-only.json": "f513ee911d5542fb4807a2644eb4fa544dce8c3fc63981ab040646b79afbc0ef",
    "r03-o2-q6-dflash2.json": "99cfda95663fff919557f31c37791a3f4784cca60ef8b33e18b2a859c6cfc5e0",
}
EXPECTED_RUNTIME_FILES = {
    "ggml.dll": "1e95ede6f763a29bf48d37e8b4ee12308d98844239099c8213e20dcb65d77fd5",
    "ggml-base.dll": "8b0fb22c6e79d8f9b80779d84e911480b64a98b69c9e391c60ab9928de21ae77",
    "ggml-cpu.dll": "7c0951b20d20453c50aed79a2eedc71adf02bdc988ccc49dea9076cead3ebb9d",
    "ggml-cuda.dll": "3893ebbe8f63c7473d49ed8efe93eb1b227f554d7b96052dfde712a84e3dc582",
    "llama.dll": "17b7f8b91a4db39af607c79e50a6f765b382a63d5cb665ce5171bf046a7c388f",
    "llama-common.dll": "cc34caba864cb9029c3c036d8068fd10d1c0cbf8af525cbbe10ff342968a3872",
    "llama-server.exe": EXPECTED_SOURCE["llama_server_sha256"],
    "llama-server-impl.dll": "32fd951a2a5115d2d36a4c90720b9766d012bed3a79c1d6d0a8529b2be5141d5",
    "mtmd.dll": "e5761e1d8d7d180a6ebe5d50a66c31d50edf5efe9619c31b9abefe1117e9105c",
}
EXPECTED_TOOLS = {
    "performance_runner_sha256": "4499101d37f8b8470a626f0f3ec69878d8b291dc7c9e772e5bb97fcef54f58d3",
    "quality_runner_sha256": "0fe653b23993fa04304f51570ebbcf480f983e2ff256d2e9f67cbdac1533b293",
    "quality_helper_sha256": "5e595a41307a6f01ab34a7deebe398814d4a8894dda2878e69c4c9c97c5e01f9",
    "evaluator_sha256": "27ee973617668f04eb359c708336b94d45920236b168a29c471155ff228827fc",
    "reporter_sha256": "5f676d00dfa16b25e1613de502d34ce379afadff28c05c583c714b964034006b",
}
EXPECTED_TARGET = {
    "file": "Qwen3.8-27B-Q6_K.gguf",
    "bytes": 22_884_408_288,
    "sha256": "562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727",
    "quantization": "Q6_K",
}
EXPECTED_DRAFT = {
    "file": "Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
    "bytes": 1_143_006_752,
    "sha256": "18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594",
    "strategy": "dflash2",
    "backend_mode": "dflash",
    "role": "auxiliary-proposer-only",
}
EXPECTED_TASK_PAIRS_SHA256 = (
    "741f100b17aa020b1d0ff5440431ea90f22cdbb21bc811595fa9970c03fd17c2"
)
EXPECTED_SECTION_DIGESTS = {
    "artifacts": "3d7b8de46470bfe97b332f55a5bf0f1fe5f71ca444f4e2eece333cd5d1a25c34",
    "hardware": "f58937d1968bd268f47c877f8e9f46bca48ab755b822ccf80d8fa05fa8e1d53e",
    "controls": "ac378f9ef2d5b4d8b7486aae24ef4659c05da6e151308e84abf48a0b8d5b4099",
    "performance": "bf18be437dfbdea24d45e9f858c7a26adad91362e514c129bc7c41a73815b165",
    "quality": "733506eeebeffd94afd8d0c13bc79850d91e52ffb41137fa45e62a6ffffa028d",
}
