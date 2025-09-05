from django.urls import path
from . import views

app_name = "aimy"

urlpatterns = [
    # Document management
    path("documents/", views.DocumentUploadView.as_view(), name="document_upload"),
    path(
        "documents/<int:document_id>/",
        views.DocumentDetailView.as_view(),
        name="document_detail",
    ),
    # path('documents/<int:document_id>/reprocess/', views.ReprocessDocumentView.as_view(), name='reprocess_document'),
    # Chat functionality
    path("chat/ask/", views.AskQuestionView.as_view(), name="ask_question"),
    path("chat/sessions/", views.ChatSessionView.as_view(), name="chat_sessions"),
    path(
        "chat/sessions/<int:session_id>/",
        views.ChatSessionDetailView.as_view(),
        name="chat_session_detail",
    ),
    path(
        "chat/sessions/<int:session_id>/messages/",
        views.ChatSessionMessagesView.as_view(),
        name="chat_session_messages",
    ),
    # Document search
    # path('documents/search/', views.DocumentSearchView.as_view(), name='document_search'),
    # Processing queue
    # path('processing-queue/', views.ProcessingQueueView.as_view(), name='processing_queue'),
    # Generic files (admin only)
    path(
        "generic-files/",
        views.GenericFileUploadView.as_view(),
        name="generic_file_upload",
    ),
    path(
        "generic-files/<int:file_id>/",
        views.GenericFileDetailView.as_view(),
        name="generic_file_detail",
    ),
    # Reminders
    path(
        "reminders/",
        views.ReminderListView.as_view(),
        name="reminder_list",
    ),
    path(
        "reminders/create/",
        views.ReminderCreateView.as_view(),
        name="reminder_create",
    ),
    path(
        "reminders/<int:reminder_id>/",
        views.ReminderDetailView.as_view(),
        name="reminder_detail",
    ),
]
