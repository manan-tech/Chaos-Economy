from typing import List, Dict, Any
from multi_agent.config import MESSAGE_HISTORY_WINDOW

class MessageChannel:
    """Supports DM (1:1), group chat, and broadcast."""

    CHANNEL_TYPES = ["dm", "group", "broadcast"]

    def __init__(self):
        self.channels: Dict[str, List[str]] = {}  # channel_id -> member list
        self.message_log: List[Dict[str, Any]] = []  # ALL messages (oversight can see)
        self.next_group_id = 0
        self.current_step = 0

    def _trim_old_messages(self) -> None:
        """Remove messages older than MESSAGE_HISTORY_WINDOW steps."""
        if self.current_step <= MESSAGE_HISTORY_WINDOW:
            return
        floor = self.current_step - MESSAGE_HISTORY_WINDOW
        self.message_log = [m for m in self.message_log if m["step"] >= floor]
    
    def send_dm(self, sender: str, recipient: str, message: str, current_step: int) -> None:
        """Private message. Only sender + recipient see it.
        BUT: oversight can subpoena the message log."""
        self.current_step = max(self.current_step, current_step)
        self.message_log.append({
            "type": "dm", "sender": sender, "recipient": recipient,
            "message": message, "step": current_step
        })
        self._trim_old_messages()
    
    def create_group(self, creator: str, members: List[str]) -> str:
        """Create a group chat. Members can coordinate."""
        channel_id = f"group_{self.next_group_id}"
        self.next_group_id += 1
        # ensure creator is in members
        if creator not in members:
            members.append(creator)
        self.channels[channel_id] = members
        return channel_id
    
    def send_group(self, sender: str, channel_id: str, message: str, current_step: int) -> None:
        """Send to group. All members see it."""
        self.current_step = max(self.current_step, current_step)
        if channel_id in self.channels and sender in self.channels[channel_id]:
            self.message_log.append({
                "type": "group", "sender": sender, "recipient": channel_id,
                "message": message, "step": current_step
            })
            self._trim_old_messages()
    
    def broadcast(self, sender: str, message: str, current_step: int) -> None:
        """Public broadcast. Everyone sees it including oversight."""
        self.current_step = max(self.current_step, current_step)
        self.message_log.append({
            "type": "broadcast", "sender": sender, "recipient": "all",
            "message": message, "step": current_step
        })
        self._trim_old_messages()

    def get_inbox(
        self, agent_id: str, current_step: int, lookback: int = 3
    ) -> List[Dict[str, Any]]:
        """Messages addressed to `agent_id` from the last `lookback` steps.

        A `lookback` of 3 means the recipient sees messages sent at
        current_step, current_step-1, and current_step-2, so a brief
        DM exchange survives a non-response step.
        """
        floor = current_step - max(0, lookback - 1)
        inbox = []
        for msg in self.message_log:
            if msg["step"] < floor or msg["step"] > current_step:
                continue
            if msg["type"] == "dm" and msg["recipient"] == agent_id:
                inbox.append(msg)
            elif msg["type"] == "group":
                channel_members = self.channels.get(msg["recipient"], [])
                if agent_id in channel_members and msg["sender"] != agent_id:
                    inbox.append(msg)
            elif msg["type"] == "broadcast" and msg["sender"] != agent_id:
                inbox.append(msg)
        return inbox
