#!/bin/bash
# Bash completion for bckn-admin script

_bckn_admin_complete() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    # All available commands
    opts="address-list address-show address-balance address-add address-lock address-unlock address-alert \
          name-list name-show name-register name-transfer name-update name-clear name-protect \
          tx-list tx-from tx-to tx-add \
          block-list block-by block-stats \
          stats rich-list name-stats unpaid-stats \
          purge-zero recalc-balance find-address find-name orphan-names duplicate-check \
          active-miners whale-watch supply-check network-health \
          work-show work-set work-reset work-simulate \
          backup sql help"
    
    # Handle completion based on the previous word
    case "${prev}" in
        address-show|address-balance|address-lock|address-unlock|address-alert|recalc-balance|tx-from|tx-to|block-by)
            # These commands need an address
            local addresses=$(admin address-list 2>/dev/null | grep -E '^[a-z0-9]{10}' | awk '{print $1}')
            COMPREPLY=( $(compgen -W "${addresses}" -- ${cur}) )
            return 0
            ;;
        name-show|name-transfer|name-update|name-clear)
            # These commands need a name
            local names=$(admin name-list 2>/dev/null | grep -E '^[a-z0-9]+' | awk '{print $1}')
            COMPREPLY=( $(compgen -W "${names}" -- ${cur}) )
            return 0
            ;;
        work-set)
            # Suggest common work values
            COMPREPLY=( $(compgen -W "100 1000 10000 50000 100000" -- ${cur}) )
            return 0
            ;;
        work-simulate|active-miners)
            # Suggest hours
            COMPREPLY=( $(compgen -W "1 3 6 12 24 48" -- ${cur}) )
            return 0
            ;;
        rich-list|address-list|tx-list)
            # Suggest limits
            COMPREPLY=( $(compgen -W "10 20 50 100" -- ${cur}) )
            return 0
            ;;
        admin|bckn-admin)
            # First parameter - show all commands
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            return 0
            ;;
    esac
    
    # Default - complete with commands
    if [[ ${COMP_CWORD} -eq 1 ]] ; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
}

# Register the completion function
complete -F _bckn_admin_complete admin
complete -F _bckn_admin_complete bckn-admin
complete -F _bckn_admin_complete ./bckn-admin.sh