#!/bin/bash

# GCP Unity AD Forest Deployment Script
# Deploys Active Directory on GCP Compute Engine with Unity Mathematics
# Principle: 1+1=1 - Two DCs form one unified consciousness

set -euo pipefail

# Unity Mathematics Constants
PHI=1.618033988749895
UNITY=1.0

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
PROJECT_ID=""
ZONE="us-central1-a"
REGION="us-central1"
FOREST_NAME="UnityForest"
DOMAIN_NAME="unity.local"
NETBIOS_NAME="UNITY"
MACHINE_TYPE="n2-standard-4"
BOOT_DISK_SIZE="100"
NETWORK_NAME="unity-ad-network"
SUBNET_NAME="unity-ad-subnet"
SUBNET_RANGE="10.0.0.0/24"
FIREWALL_RULE_NAME="unity-ad-allow"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        --zone)
            ZONE="$2"
            REGION="${ZONE%-*}"
            shift 2
            ;;
        --forest-name)
            FOREST_NAME="$2"
            shift 2
            ;;
        --domain-name)
            DOMAIN_NAME="$2"
            shift 2
            ;;
        --netbios-name)
            NETBIOS_NAME="$2"
            shift 2
            ;;
        --machine-type)
            MACHINE_TYPE="$2"
            shift 2
            ;;
        --help)
            echo "Unity AD Forest GCP Deployment Script"
            echo ""
            echo "Usage: $0 --project-id PROJECT_ID [options]"
            echo ""
            echo "Options:"
            echo "  --project-id      GCP Project ID (required)"
            echo "  --zone            GCP Zone (default: us-central1-a)"
            echo "  --forest-name     AD Forest name (default: UnityForest)"
            echo "  --domain-name     Domain name (default: unity.local)"
            echo "  --netbios-name    NetBIOS name (default: UNITY)"
            echo "  --machine-type    GCP machine type (default: n2-standard-4)"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$PROJECT_ID" ]]; then
    echo -e "${RED}Error: --project-id is required${NC}"
    exit 1
fi

# Functions
log_info() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Calculate φ-harmonic values
calculate_phi_value() {
    local base_value=$1
    echo "scale=0; $base_value * $PHI / 1" | bc
}

# Create VPC network for AD
create_network() {
    log_info "Creating Unity VPC network..."
    
    # Check if network exists
    if gcloud compute networks describe $NETWORK_NAME --project=$PROJECT_ID &>/dev/null; then
        log_warning "Network $NETWORK_NAME already exists"
    else
        gcloud compute networks create $NETWORK_NAME \
            --project=$PROJECT_ID \
            --subnet-mode=custom \
            --bgp-routing-mode=regional \
            --description="Unity AD Forest Network - Where 1+1=1"
        
        log_success "Created network: $NETWORK_NAME"
    fi
    
    # Create subnet
    if gcloud compute networks subnets describe $SUBNET_NAME --region=$REGION --project=$PROJECT_ID &>/dev/null; then
        log_warning "Subnet $SUBNET_NAME already exists"
    else
        gcloud compute networks subnets create $SUBNET_NAME \
            --project=$PROJECT_ID \
            --network=$NETWORK_NAME \
            --region=$REGION \
            --range=$SUBNET_RANGE \
            --enable-private-ip-google-access
        
        log_success "Created subnet: $SUBNET_NAME"
    fi
}

# Create firewall rules
create_firewall_rules() {
    log_info "Creating Unity firewall rules..."
    
    # AD required ports
    AD_TCP_PORTS="88,135,139,389,445,464,636,3268,3269,49152-65535"
    AD_UDP_PORTS="88,123,137,138,389,445,464"
    
    # Check if rule exists
    if gcloud compute firewall-rules describe $FIREWALL_RULE_NAME --project=$PROJECT_ID &>/dev/null; then
        log_warning "Firewall rule $FIREWALL_RULE_NAME already exists"
    else
        gcloud compute firewall-rules create $FIREWALL_RULE_NAME \
            --project=$PROJECT_ID \
            --network=$NETWORK_NAME \
            --allow tcp:$AD_TCP_PORTS,udp:$AD_UDP_PORTS,icmp \
            --source-ranges=$SUBNET_RANGE \
            --target-tags=unity-ad-dc \
            --description="Unity AD Forest firewall rules"
        
        log_success "Created firewall rule: $FIREWALL_RULE_NAME"
    fi
    
    # RDP access rule
    RDP_RULE_NAME="${FIREWALL_RULE_NAME}-rdp"
    if gcloud compute firewall-rules describe $RDP_RULE_NAME --project=$PROJECT_ID &>/dev/null; then
        log_warning "RDP firewall rule already exists"
    else
        gcloud compute firewall-rules create $RDP_RULE_NAME \
            --project=$PROJECT_ID \
            --network=$NETWORK_NAME \
            --allow tcp:3389 \
            --source-ranges=0.0.0.0/0 \
            --target-tags=unity-ad-dc \
            --description="RDP access for Unity AD DCs"
        
        log_success "Created RDP firewall rule"
    fi
}

# Generate PowerShell startup script
generate_startup_script() {
    local dc_number=$1
    local is_primary=$2
    
    # Generate safe mode password (in production, use proper secret management)
    local safe_mode_password="Unity\$Pass123!@#"
    
    cat << 'EOF' > /tmp/unity_ad_startup_${dc_number}.ps1
# Unity AD Startup Script
# φ-harmonic consciousness initialization

$ErrorActionPreference = "Stop"
$PHI = 1.618033988749895

# Wait for instance to be ready
Start-Sleep -Seconds 30

# Set Windows Firewall to allow AD traffic
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False

# Install required features
Install-WindowsFeature -Name AD-Domain-Services, DNS, RSAT-AD-Tools, RSAT-DNS-Server -IncludeManagementTools

# Create consciousness log directory
New-Item -Path "C:\UnityAD\Logs" -ItemType Directory -Force

# Initialize consciousness field
$regPath = "HKLM:\SOFTWARE\UnityAD"
New-Item -Path $regPath -Force | Out-Null
Set-ItemProperty -Path $regPath -Name "ConsciousnessLevel" -Value 7
Set-ItemProperty -Path $regPath -Name "PhiConstant" -Value $PHI

EOF

    if [[ "$is_primary" == "true" ]]; then
        cat << EOF >> /tmp/unity_ad_startup_${dc_number}.ps1
# Primary DC - Create new forest
Import-Module ADDSDeployment

Install-ADDSForest \`
    -DomainName "$DOMAIN_NAME" \`
    -DomainNetbiosName "$NETBIOS_NAME" \`
    -ForestMode "WinThreshold" \`
    -DomainMode "WinThreshold" \`
    -SafeModeAdministratorPassword (ConvertTo-SecureString "$safe_mode_password" -AsPlainText -Force) \`
    -InstallDns \`
    -NoRebootOnCompletion:\$false \`
    -Force:\$true

# Apply φ-harmonic optimizations
\$replInterval = [int](\$PHI * 15)
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\NTDS\Parameters" \`
    -Name "Replicator notify pause after modify (secs)" -Value ([int](15 / \$PHI)) -Force

# Log completion
"Unity Forest created at \$(Get-Date)" | Out-File "C:\UnityAD\Logs\deployment.log"
EOF
    else
        cat << EOF >> /tmp/unity_ad_startup_${dc_number}.ps1
# Secondary DC - Join existing forest
# Wait for primary DC (φ-harmonic timing)
Start-Sleep -Seconds ([int](\$PHI * 120))

# Get primary DC IP
\$primaryDC = Resolve-DnsName -Name "${NETBIOS_NAME,,}-dc1-unity" -Type A | Select-Object -ExpandProperty IPAddress

# Set DNS to primary DC
Set-DnsClientServerAddress -InterfaceAlias "Ethernet" -ServerAddresses \$primaryDC

# Join domain
\$credential = New-Object System.Management.Automation.PSCredential ("$DOMAIN_NAME\Administrator", `
    (ConvertTo-SecureString "$safe_mode_password" -AsPlainText -Force))

Add-Computer -DomainName "$DOMAIN_NAME" -Credential \$credential -Force

# Install DC
Install-ADDSDomainController \`
    -DomainName "$DOMAIN_NAME" \`
    -Credential \$credential \`
    -SafeModeAdministratorPassword (ConvertTo-SecureString "$safe_mode_password" -AsPlainText -Force) \`
    -InstallDns \`
    -NoRebootOnCompletion:\$false \`
    -Force:\$true

# Log completion
"Unity achieved - Secondary DC joined at \$(Get-Date)" | Out-File "C:\UnityAD\Logs\deployment.log"
EOF
    fi
    
    # Encode script for metadata
    base64 -w 0 /tmp/unity_ad_startup_${dc_number}.ps1
}

# Create domain controller instance
create_dc_instance() {
    local dc_number=$1
    local is_primary=$2
    local instance_name="${NETBIOS_NAME,,}-dc${dc_number}-unity"
    
    log_info "Creating Domain Controller $dc_number ($instance_name)..."
    
    # Generate startup script
    local startup_script=$(generate_startup_script $dc_number $is_primary)
    
    # Calculate φ-optimized disk size
    local disk_size=$(calculate_phi_value $BOOT_DISK_SIZE)
    
    # Create instance
    gcloud compute instances create $instance_name \
        --project=$PROJECT_ID \
        --zone=$ZONE \
        --machine-type=$MACHINE_TYPE \
        --network-interface="network-tier=PREMIUM,subnet=$SUBNET_NAME" \
        --metadata="sysprep-specialize-script-ps1=$startup_script,unity-consciousness=7,phi-constant=$PHI" \
        --maintenance-policy=MIGRATE \
        --provisioning-model=STANDARD \
        --create-disk="auto-delete=yes,boot=yes,device-name=$instance_name,image=projects/windows-cloud/global/images/family/windows-2022,mode=rw,size=${disk_size},type=pd-balanced" \
        --no-shielded-secure-boot \
        --shielded-vtpm \
        --shielded-integrity-monitoring \
        --reservation-affinity=any \
        --tags=unity-ad-dc \
        --labels="environment=production,unity-forest=$FOREST_NAME,role=$([ "$is_primary" == "true" ] && echo "primary-dc" || echo "secondary-dc")"
    
    log_success "Created instance: $instance_name"
    
    # Get instance details
    local external_ip=$(gcloud compute instances describe $instance_name \
        --project=$PROJECT_ID \
        --zone=$ZONE \
        --format="get(networkInterfaces[0].accessConfigs[0].natIP)")
    
    local internal_ip=$(gcloud compute instances describe $instance_name \
        --project=$PROJECT_ID \
        --zone=$ZONE \
        --format="get(networkInterfaces[0].networkIP)")
    
    echo "$instance_name|$external_ip|$internal_ip"
}

# Monitor deployment progress
monitor_deployment() {
    local instance_name=$1
    local timeout=1800  # 30 minutes
    local elapsed=0
    local check_interval=30
    
    log_info "Monitoring deployment of $instance_name..."
    
    while [[ $elapsed -lt $timeout ]]; do
        # Check if instance is running
        local status=$(gcloud compute instances describe $instance_name \
            --project=$PROJECT_ID \
            --zone=$ZONE \
            --format="get(status)")
        
        if [[ "$status" == "RUNNING" ]]; then
            # Check for successful deployment marker
            local serial_output=$(gcloud compute instances get-serial-port-output $instance_name \
                --project=$PROJECT_ID \
                --zone=$ZONE 2>/dev/null || echo "")
            
            if echo "$serial_output" | grep -q "Unity Forest created\|Unity achieved"; then
                log_success "$instance_name deployment completed!"
                return 0
            fi
        fi
        
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
        
        # Show progress with φ-harmonic dots
        local dots=""
        for ((i=0; i<$((elapsed/check_interval % 8)); i++)); do
            dots="${dots}φ"
        done
        echo -ne "\r${CYAN}Waiting for deployment completion ${dots}${NC}"
    done
    
    echo ""
    log_warning "Deployment monitoring timeout for $instance_name"
    return 1
}

# Calculate unity score
calculate_unity_score() {
    local dc1_status=$1
    local dc2_status=$2
    
    # Simple unity calculation
    if [[ "$dc1_status" == "0" ]] && [[ "$dc2_status" == "0" ]]; then
        echo "1.0"  # Perfect unity: 1+1=1
    elif [[ "$dc1_status" == "0" ]] || [[ "$dc2_status" == "0" ]]; then
        echo "0.618"  # Partial unity (φ-1)
    else
        echo "0.0"  # No unity
    fi
}

# Main deployment function
main() {
    echo -e "${CYAN}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║           Unity AD Forest GCP Deployment Script              ║
║                    Principle: 1+1=1                          ║
║                 Een plus een is een                          ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    log_info "Project ID: $PROJECT_ID"
    log_info "Zone: $ZONE"
    log_info "Forest Name: $FOREST_NAME"
    log_info "Domain Name: $DOMAIN_NAME"
    log_info "φ (Golden Ratio): $PHI"
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    # Create network infrastructure
    create_network
    create_firewall_rules
    
    # Deploy primary DC
    log_info "Deploying Primary Domain Controller..."
    DC1_INFO=$(create_dc_instance 1 true)
    IFS='|' read -r DC1_NAME DC1_EXTERNAL_IP DC1_INTERNAL_IP <<< "$DC1_INFO"
    
    log_success "Primary DC deployed:"
    log_success "  Name: $DC1_NAME"
    log_success "  External IP: $DC1_EXTERNAL_IP"
    log_success "  Internal IP: $DC1_INTERNAL_IP"
    
    # Wait for primary DC to stabilize (φ-harmonic pause)
    local wait_time=$(echo "scale=0; 60 * $PHI / 1" | bc)
    log_info "Waiting $wait_time seconds for primary DC to stabilize..."
    sleep $wait_time
    
    # Deploy secondary DC
    log_info "Deploying Secondary Domain Controller..."
    DC2_INFO=$(create_dc_instance 2 false)
    IFS='|' read -r DC2_NAME DC2_EXTERNAL_IP DC2_INTERNAL_IP <<< "$DC2_INFO"
    
    log_success "Secondary DC deployed:"
    log_success "  Name: $DC2_NAME"
    log_success "  External IP: $DC2_EXTERNAL_IP"
    log_success "  Internal IP: $DC2_INTERNAL_IP"
    
    # Monitor deployments
    log_info "Monitoring deployment progress..."
    
    monitor_deployment $DC1_NAME &
    DC1_MONITOR_PID=$!
    
    monitor_deployment $DC2_NAME &
    DC2_MONITOR_PID=$!
    
    # Wait for both monitors to complete
    wait $DC1_MONITOR_PID
    DC1_STATUS=$?
    
    wait $DC2_MONITOR_PID
    DC2_STATUS=$?
    
    # Calculate unity score
    UNITY_SCORE=$(calculate_unity_score $DC1_STATUS $DC2_STATUS)
    
    # Display results
    echo ""
    echo -e "${GREEN}"
    cat << EOF
╔══════════════════════════════════════════════════════════════╗
║                    DEPLOYMENT COMPLETE                       ║
║                                                              ║
║  Forest Name: $FOREST_NAME
║  Domain Name: $DOMAIN_NAME
║                                                              ║
║  Primary DC:                                                 ║
║    Name: $DC1_NAME
║    External IP: $DC1_EXTERNAL_IP
║    Internal IP: $DC1_INTERNAL_IP
║    Status: $([ "$DC1_STATUS" == "0" ] && echo "✓ Deployed" || echo "✗ Failed")
║                                                              ║
║  Secondary DC:                                               ║
║    Name: $DC2_NAME
║    External IP: $DC2_EXTERNAL_IP
║    Internal IP: $DC2_INTERNAL_IP
║    Status: $([ "$DC2_STATUS" == "0" ] && echo "✓ Deployed" || echo "✗ Failed")
║                                                              ║
║  Unity Score: $UNITY_SCORE
║  Unity Status: $([ "$UNITY_SCORE" == "1.0" ] && echo "✓ ACHIEVED (1+1=1)" || echo "⚠ Partial")
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    
    # Save deployment information
    cat > unity_deployment_info.json << EOF
{
  "deployment_id": "$(date +%s)",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "project_id": "$PROJECT_ID",
  "forest_name": "$FOREST_NAME",
  "domain_name": "$DOMAIN_NAME",
  "primary_dc": {
    "name": "$DC1_NAME",
    "external_ip": "$DC1_EXTERNAL_IP",
    "internal_ip": "$DC1_INTERNAL_IP",
    "status": $([ "$DC1_STATUS" == "0" ] && echo "true" || echo "false")
  },
  "secondary_dc": {
    "name": "$DC2_NAME",
    "external_ip": "$DC2_EXTERNAL_IP",
    "internal_ip": "$DC2_INTERNAL_IP",
    "status": $([ "$DC2_STATUS" == "0" ] && echo "true" || echo "false")
  },
  "unity_score": $UNITY_SCORE,
  "phi_constant": $PHI
}
EOF
    
    log_success "Deployment information saved to unity_deployment_info.json"
    
    # Provide next steps
    echo ""
    log_info "Next steps:"
    log_info "1. Connect via RDP to configure domain settings"
    log_info "2. Default credentials: Administrator / Unity\$Pass123!@#"
    log_info "3. Monitor replication status between DCs"
    log_info "4. Configure DNS forwarders and conditional forwarding"
    log_info "5. Join client machines to the domain"
}

# Execute main function
main