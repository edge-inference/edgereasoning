for dir in results/*/; do
    echo "=== $(basename "$dir") ==="
    find "$dir" -name "*.*" -type f | sed 's/.*\.//' | sort | uniq -c
    echo
done

