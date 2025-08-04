import user_agents
from accounts.models import AccessGuardian


def log_access_guardian(request, log_type, phone_number=""):
    try:
        x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
        user_agent_string = request.META.get("HTTP_USER_AGENT", "")
        channel = request.META.get("HTTP_CHANNEL", "")
        agent = user_agents.parse(user_agent_string)

        if not channel:
            channel = (
                "Mobile"
                if agent.is_mobile
                else (
                    "Tablet" if agent.is_tablet else "PC" if agent.is_pc else "Unknown"
                )
            )

        if x_forwarded_for:
            ip = x_forwarded_for.split(",")[0]
        else:
            ip = request.META.get("REMOTE_ADDR")
        AccessGuardian.objects.create(
            log_type=log_type,
            phone_number=phone_number,
            device=channel,
            browser=agent.browser.family,
            browser_version=agent.browser.version_string,
            os=agent.os.family,
            os_version=agent.os.version_string,
            ip_address=ip,
        )
    except Exception:
        pass
