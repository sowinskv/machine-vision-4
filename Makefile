.PHONY: train train-bg train-safe test monitor stop status clean diagram help

help:
	@echo "CIFAR-10 CNN Training"
	@echo ""
	@echo "make train          run training (foreground)"
	@echo "make train-bg       run training in background with logs"
	@echo "make train-safe     background training + prevent Mac sleep"
	@echo "make test           quick test run (2 epochs)"
	@echo "make monitor        watch training progress"
	@echo "make status         check if training is running"
	@echo "make stop           stop training process"
	@echo "make diagram        generate architecture diagram"
	@echo "make clean          remove outputs and logs"
	@echo ""

train:
	uv run python main.py

train-bg:
	@echo "starting training in background..."
	@nohup bash -c "uv run python main.py > training.log 2>&1" &
	@sleep 2
	@echo "training started. monitor with: make monitor"
	@echo "check status with: make status"

train-safe:
	@echo "starting training with sleep prevention..."
	@nohup bash -c "uv run python main.py > training.log 2>&1" &
	@sleep 3
	@TRAIN_PID=$$(pgrep -f "python.*main.py" | tail -1); \
	if [ -n "$$TRAIN_PID" ]; then \
		caffeinate -i -w $$TRAIN_PID & \
		echo "✓ training started (PID: $$TRAIN_PID)"; \
		echo "✓ sleep prevention active"; \
		echo ""; \
		echo "monitor: make monitor"; \
		echo "status:  make status"; \
	else \
		echo "✗ training failed to start"; \
	fi

test:
	@echo "running quick test (2 epochs)..."
	@sed -i.bak 's/EPOCHS = 150/EPOCHS = 2/' main.py
	@uv run python main.py
	@mv main.py.bak main.py
	@echo "✓ test complete, EPOCHS restored to 150"

monitor:
	tail -f training.log

status:
	@echo "checking training status..."
	@TRAIN_PID=$$(pgrep -f "python.*main.py" | grep -v grep); \
	if [ -n "$$TRAIN_PID" ]; then \
		echo "✓ training is running (PID: $$TRAIN_PID)"; \
		CAFE_PID=$$(pgrep -f "caffeinate.*$$TRAIN_PID"); \
		if [ -n "$$CAFE_PID" ]; then \
			echo "✓ sleep prevention active (PID: $$CAFE_PID)"; \
		else \
			echo "⚠ sleep prevention not active"; \
		fi; \
		if [ -f training.log ]; then \
			echo ""; \
			echo "last log entry:"; \
			tail -1 training.log; \
		fi; \
	else \
		echo "✗ training is not running"; \
		if [ -f training.log ]; then \
			echo ""; \
			echo "last log entry:"; \
			tail -1 training.log; \
		fi; \
	fi

stop:
	@echo "stopping training..."
	@pkill -f "python.*main.py" && echo "✓ training stopped" || echo "✗ no training process found"
	@pkill caffeinate && echo "✓ sleep prevention stopped" || true

diagram:
	uv run python generate_architecture_diagram.py

clean:
	@echo "cleaning outputs..."
	rm -rf outputs/models/*.keras
	rm -rf outputs/plots/*.png
	rm -rf outputs/reports/*.json
	rm -f training.log
	rm -f nohup.out
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ clean complete"
