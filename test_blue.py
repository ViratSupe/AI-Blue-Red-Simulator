from agents.blue_agent import BlueTeamAgent

agent = BlueTeamAgent()

test_log = {
    "log_id": "test123",
    "attack_type": "brute_force",
    "source_ip": "192.168.1.100",
    "timestamp": "2026-04-08T10:00:00Z",
    "payload": "admin login failed multiple times",
    "attempts": 200
}

result = agent.analyse(test_log)

print("\n===== RESULT =====")
print(result)