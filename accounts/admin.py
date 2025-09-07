from django.contrib import admin
from django.utils.html import format_html
from django.db.models import Q
from django.contrib import messages
from unfold.admin import ModelAdmin
from unfold.forms import AdminPasswordChangeForm, UserChangeForm, UserCreationForm
from .models import CustomUser, College, Department


@admin.register(College)
class CollegeAdmin(ModelAdmin):
    list_display = ["name", "department_count", "created_at", "updated_at"]
    search_fields = ["name", "description"]
    readonly_fields = ["created_at", "updated_at"]
    ordering = ["name"]
    # Unfold specific configurations
    list_display_links = ["name"]
    list_per_page = 20
    show_full_result_count = False

    def department_count(self, obj):
        count = obj.departments.count()
        return format_html('<span class="badge badge-primary">{}</span>', count)

    department_count.short_description = "Departments"
    department_count.admin_order_field = "departments__count"

    fieldsets = (
        (
            "Basic Information",
            {
                "fields": ("name", "description"),
                "classes": ("unfold-sections-section",),
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at"),
                "classes": ("unfold-sections-section", "collapse"),
            },
        ),
    )


@admin.register(Department)
class DepartmentAdmin(ModelAdmin):
    list_display = ["name", "college", "user_count", "created_at"]
    list_filter = ["college"]
    search_fields = ["name", "description", "college__name"]
    readonly_fields = ["created_at", "updated_at"]
    ordering = ["college__name", "name"]

    # Unfold specific configurations
    list_display_links = ["name"]
    list_per_page = 20
    show_full_result_count = False

    def user_count(self, obj):
        count = obj.users.count()
        return format_html('<span class="badge badge-success">{}</span>', count)

    user_count.short_description = "Students"
    user_count.admin_order_field = "users__count"

    fieldsets = (
        (
            "Basic Information",
            {
                "fields": ("name", "description", "college"),
                "classes": ("unfold-sections-section",),
            },
        ),
        (
            "Timestamps",
            {
                "fields": ("created_at", "updated_at"),
                "classes": ("unfold-sections-section", "collapse"),
            },
        ),
    )


class CustomUserAdmin(ModelAdmin):
    form = UserChangeForm
    add_form = UserCreationForm
    change_password_form = AdminPasswordChangeForm

    # Enhanced list display with Unfold styling
    list_display = [
        "email",
        "full_name",
        "student_id",
        "phone_number",
        "department",
        "year_of_study",
        "is_active",
        "is_staff",
        "is_superuser",
        "last_login",
        "date_joined",
        "status_badge",
    ]

    # Advanced filters
    list_filter = [
        "is_active",
        "is_staff",
        "is_superuser",
        "department__college",
        "department",
        "year_of_study",
        "groups",
        "date_joined",
        "last_login",
    ]

    # Search functionality
    search_fields = [
        "email",
        "username",
        "first_name",
        "last_name",
        "student_id",
        "index_number",
        "phone_number",
        "department__name",
        "department__college__name",
    ]

    # Readonly fields
    readonly_fields = ["date_joined", "last_login"]

    # Ordering
    ordering = ["-date_joined"]

    # Items per page
    list_per_page = 25

    # Actions
    actions = [
        "activate_users",
        "deactivate_users",
        "send_welcome_email",
        "export_user_data",
    ]

    # Fieldsets for better organization - using standard Django UserAdmin structure
    fieldsets = (
        (None, {"fields": ("username", "password")}),
        (
            "Personal info",
            {
                "fields": (
                    "first_name",
                    "last_name",
                    "email",
                    "phone_number",
                    "profile_image",
                )
            },
        ),
        (
            "Academic Information",
            {
                "fields": ("student_id", "index_number", "department", "year_of_study"),
                "classes": ("unfold-sections-section",),
            },
        ),
        (
            "Permissions",
            {
                "fields": (
                    "is_active",
                    "is_staff",
                    "is_superuser",
                    "groups",
                    "user_permissions",
                ),
                "classes": ("unfold-sections-section",),
            },
        ),
        (
            "Important dates",
            {
                "fields": ("last_login", "date_joined"),
                "classes": ("unfold-sections-section", "collapse"),
            },
        ),
    )

    # Add fieldsets
    add_fieldsets = (
        (
            None,
            {
                "classes": ("wide",),
                "fields": (
                    "username",
                    "email",
                    "password1",
                    "password2",
                    "first_name",
                    "last_name",
                    "student_id",
                    "index_number",
                    "department",
                    "year_of_study",
                    "profile_image",
                ),
            },
        ),
    )

    # Custom methods with Unfold styling
    def full_name(self, obj):
        name = f"{obj.first_name} {obj.last_name}".strip() or "N/A"
        return format_html('<span class="font-medium text-gray-900">{}</span>', name)

    full_name.short_description = "Full Name"
    full_name.admin_order_field = "first_name"

    def status_badge(self, obj):
        if obj.is_active:
            return format_html(
                '<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">✓ Active</span>'
            )
        else:
            return format_html(
                '<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">✗ Inactive</span>'
            )

    status_badge.short_description = "Status"

    def student_id_display(self, obj):
        return format_html(
            '<span class="font-mono text-sm bg-gray-100 px-2 py-1 rounded">{}</span>',
            obj.student_id,
        )

    student_id_display.short_description = "Student ID"

    def year_of_study_display(self, obj):
        return format_html(
            '<span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">Level {}</span>',
            obj.year_of_study,
        )

    year_of_study_display.short_description = "Year of Study"

    # Admin actions
    def activate_users(self, request, queryset):
        updated = queryset.update(is_active=True)
        self.message_user(
            request, f"Successfully activated {updated} user(s).", messages.SUCCESS
        )

    activate_users.short_description = "Activate selected users"

    def deactivate_users(self, request, queryset):
        updated = queryset.update(is_active=False)
        self.message_user(
            request, f"Successfully deactivated {updated} user(s).", messages.SUCCESS
        )

    deactivate_users.short_description = "Deactivate selected users"

    def send_welcome_email(self, request, queryset):
        count = queryset.count()
        self.message_user(
            request, f"Welcome email would be sent to {count} user(s).", messages.INFO
        )

    send_welcome_email.short_description = "Send welcome email to selected users"

    def export_user_data(self, request, queryset):
        count = queryset.count()
        self.message_user(
            request,
            f"Export data for {count} user(s) would be generated.",
            messages.INFO,
        )

    export_user_data.short_description = "Export selected users data"

    # Custom admin methods
    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related("department", "department__college")

    def get_search_results(self, request, queryset, search_term):
        queryset, use_distinct = super().get_search_results(
            request, queryset, search_term
        )

        if search_term:
            queryset |= self.model.objects.filter(
                Q(department__name__icontains=search_term)
                | Q(department__college__name__icontains=search_term)
            )

        return queryset, use_distinct


# Register the enhanced admin
admin.site.register(CustomUser, CustomUserAdmin)


# Customize admin site with Unfold branding
admin.site.site_header = "Aimy AI Administration"
admin.site.site_title = "Aimy AI Admin"
admin.site.index_title = "Welcome to Aimy AI Administration"


# Add custom admin actions for the entire site
@admin.action(description="Mark selected items as active")
def make_active(modeladmin, request, queryset):
    if hasattr(queryset.model, "is_active"):
        updated = queryset.update(is_active=True)
        modeladmin.message_user(
            request,
            f"{updated} items were successfully marked as active.",
            messages.SUCCESS,
        )


@admin.action(description="Mark selected items as inactive")
def make_inactive(modeladmin, request, queryset):
    if hasattr(queryset.model, "is_active"):
        updated = queryset.update(is_active=False)
        modeladmin.message_user(
            request,
            f"{updated} items were successfully marked as inactive.",
            messages.SUCCESS,
        )
