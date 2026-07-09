async function addItem() {
    const item = document.getElementById("itemInput").value;

    await fetch("/add", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ item: item }),
    });

    location.reload();
}
