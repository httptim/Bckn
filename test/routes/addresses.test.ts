/*
 * Copyright 2016 - 2024 Drew Edwards, tmpim
 *
 * This file is part of Krist.
 *
 * Krist is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Krist is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Krist. If not, see <http://www.gnu.org/licenses/>.
 *
 * For more project information, see <https://github.com/tmpim/krist>.
 */

import { expect } from "chai";
import { api } from "../api.js";
import { seed } from "../seed.js";

describe("v1 routes: addresses", function() {
  before(seed);

  describe("GET /?getbalance", function() {
    it("should return the balance", async function() {
      const res = await api().get("/?getbalance=bv8c78oh67");
      expect(res).to.have.status(200);
      expect(res).to.be.text;
      expect(res.text).to.equal("10");
    });

    it("should return the balance", async function() {
      const res = await api().get("/?getbalance=bu1sdlbvwh");
      expect(res).to.have.status(200);
      expect(res).to.be.text;
      expect(res.text).to.equal("0");
    });

    it("should return 0 for a non-existent address", async function() {
      const res = await api().get("/?getbalance=bnotfound0");
      expect(res).to.have.status(200);
      expect(res).to.be.text;
      expect(res.text).to.equal("0");
    });
  });

  describe("GET /?alert", function() {
    it("should return an alert", async function() {
      const res = await api().get("/?alert=c");
      expect(res).to.have.status(200);
      expect(res).to.be.text;
      expect(res.text).to.equal("Test alert");
    });

    it("should return nothing for addresses with no alert", async function() {
      const res = await api().get("/?alert=a");
      expect(res).to.have.status(200);
      expect(res).to.be.text;
      expect(res.text).to.equal("");
    });
  });

  // TODO: GET /?richapi
  // TODO: GET /?listtx
});

describe("v2 routes: addresses", function() {
  // TODO: GET /addresses

  describe("POST /addresses/alert", function() {
    it("should return an alert", async function() {
      const res = await api()
        .post("/addresses/alert")
        .send({ privatekey: "c" });

      expect(res).to.have.status(200);
      expect(res).to.be.json;
      expect(res.body).to.deep.equal({ ok: true, alert: "Test alert" });
    });

    it("should return null for addresses with no alert", async function() {
      const res = await api()
        .post("/addresses/alert")
        .send({ privatekey: "a" });

      expect(res).to.have.status(200);
      expect(res).to.be.json;
      expect(res.body).to.deep.equal({ ok: true, alert: null });
    });

    it("should return an error for addresses that doesn't exist", async function() {
      const res = await api()
        .post("/addresses/alert")
        .send({ privatekey: "notfound" });

      expect(res).to.be.json;
      expect(res.body).to.deep.include({ ok: false, error: "address_not_found" });
    });
  });

  // TODO: GET /addresses/rich

  describe("GET /addresses/:address", function() {
    it("should get an address", async function() {
      const res = await api().get("/addresses/bv8c78oh67");
      expect(res).to.have.status(200);
      expect(res).to.be.json;
      expect(res.body).to.include({ ok: true });
      expect(res.body.address).to.be.an("object");
      expect(res.body.address).to.have.all.keys("address", "balance", "totalin", "totalout", "firstseen");
      expect(res.body.address).to.not.have.key("names");
    });

    it("should get an address with names", async function() {
      const res = await api().get("/addresses/bv8c78oh67?fetchNames");
      expect(res).to.be.json;
      expect(res.body).to.include({ ok: true });
      expect(res.body.address).to.be.an("object");
      expect(res.body.address.names).to.equal(0);
    });

    it("should not contain private parts", async function() {
      const res = await api().get("/addresses/byouf00c9w");
      expect(res).to.have.status(200);
      expect(res).to.be.json;
      expect(res.body.address).to.be.an("object");
      expect(res.body.address).to.not.have.any.keys("id", "privatekey", "alert", "locked");
    });

    it("should return an error for addresses that doesn't exist", async function() {
      const res = await api().get("/addresses/bnotfound0");
      expect(res).to.be.json;
      expect(res.body).to.deep.include({ ok: false, error: "address_not_found" });
    });
  });

  // TODO: GET /addresses/:address/names
  // TODO: GET /addresses/:address/transactions
});
